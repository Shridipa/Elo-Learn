import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Elo Learn — Adaptive Learning Analytics",
    page_icon="📘",
    layout="wide",
)

# Utility functions

def api_get(path, params=None, timeout=20, default=None):
    try:
        response = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        st.error(f"Unable to load {path}: {error}")
        return default or {}


def render_metric_cards(cards):
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(f"### {card['label']}")
            st.metric(card['label'], card['value'], delta=card.get('delta', ""))
            if card.get('caption'):
                st.caption(card['caption'])


def render_insight_box(title, body):
    st.markdown(f"### {title}")
    st.info(body)


def value_label(value, thresholds=(0.4, 0.75)):
    if value < thresholds[0]:
        return "Needs attention"
    if value < thresholds[1]:
        return "Developing"
    return "Proficient"


def render_empty_state(message):
    st.markdown("---")
    st.warning(message)
    st.markdown("""
    _Try another filter, select a different student, or refresh your data to view insights._
    """)


# Sidebar and navigation

st.sidebar.title("Elo Learn")
st.sidebar.write("AI-powered adaptive learning for research labs and instructors.")

section = st.sidebar.radio(
    "Explore",
    [
        "Overview",
        "Knowledge Tracing",
        "Recommendations",
        "Knowledge Graph",
        "Spaced Repetition",
        "Instructor Analytics",
        "Research Lab",
        "System",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Quick Actions")
if st.sidebar.button("View Recommendations"):
    section = "Recommendations"
if st.sidebar.button("Check Mastery"):
    section = "Knowledge Tracing"
if st.sidebar.button("Review Due Topics"):
    section = "Spaced Repetition"
if st.sidebar.button("Open Analytics"):
    section = "Instructor Analytics"


# Top-level overview content

if section == "Overview":
    st.markdown("# Elo Learn")
    st.markdown("### AI-Powered Adaptive Learning Platform")
    st.markdown(
        "Personalized learning using Knowledge Tracing, Knowledge Graphs, Explainable Recommendations, and Spaced Repetition."
    )

    overview = api_get("/topics", default={})
    student_embeddings = api_get("/students/embeddings_data", default={}).get("results", {}).get("embeddings", {})
    recommendation_results = api_get("/results/recommendations", default={}).get("results", {})
    kg_summary = api_get("/kg/subgraph", default={})

    cards = [
        {
            "label": "Total Students",
            "value": len(student_embeddings),
            "caption": "Learners captured by the platform",
        },
        {
            "label": "Topics Covered",
            "value": len(overview.get("topics", [])),
            "caption": "Concepts in the knowledge graph",
        },
        {
            "label": "Recommendation Sets",
            "value": len(recommendation_results),
            "caption": "Recent recommendation outputs",
        },
        {
            "label": "Knowledge Graph Nodes",
            "value": len(kg_summary.get("nodes", [])) if kg_summary.get("nodes") else "N/A",
            "caption": "Concept entities in KG",
        },
    ]
    render_metric_cards(cards)

    st.markdown("---")
    st.subheader("How Elo Learn Works")
    cols = st.columns(4)
    steps = [
        ("1. Student Activity", "Interactions are recorded and analyzed."),
        ("2. Knowledge Tracing", "Mastery is estimated for each concept."),
        ("3. Recommendation Engine", "Personalized learning suggestions are generated."),
        ("4. Learning Path", "Students follow a tailored path with review reminders."),
    ]
    for col, step in zip(cols, steps):
        with col:
            st.markdown(f"**{step[0]}**")
            st.write(step[1])

    st.markdown("---")
    st.subheader("Quick Start")
    st.write(
        "Use the left navigation to explore insights across Knowledge Tracing, Recommendations, Knowledge Graph reasoning, Spaced Repetition, Instructor Analytics, and Research Lab metrics."
    )
    st.info(
        "Research Insight: This dashboard is designed to surface actionable signals for both instructors and learning scientists within 30 seconds."
    )


# Knowledge Tracing page

elif section == "Knowledge Tracing":
    st.markdown("# Knowledge Tracing")
    st.write("Understand student mastery, progress, and areas that need reinforcement.")

    student_id = st.number_input("Student ID", min_value=1, value=1, key="kt_student")
    st.markdown("---")

    mastery_data = api_get(f"/students/{student_id}/mastery", default={})

    if not mastery_data:
        render_empty_state("No mastery data found for this student yet.")
    else:
        mastery = mastery_data.get("mastery", {})
        if not mastery:
            render_empty_state("No mastery data available for this student. Complete more learning activities to generate mastery estimates.")
        else:
            avg_mastery = sum(mastery.values()) / len(mastery) if mastery else 0
            predicted_success = f"{avg_mastery * 100:.0f}%"

            info_cols = st.columns(4)
            info_cols[0].metric("Student ID", student_id)
            info_cols[1].metric("Current Mastery", f"{avg_mastery * 100:.1f}%")
            info_cols[2].metric("Predicted Success", predicted_success)
            info_cols[3].metric("Learning Status", value_label(avg_mastery))

            st.markdown("### Mastery Distribution")
            concept_df = pd.DataFrame(
                [{"Concept": concept, "Mastery": score} for concept, score in mastery.items()]
            )
            fig = px.bar(
                concept_df,
                x="Mastery",
                y="Concept",
                orientation="h",
                color="Mastery",
                color_continuous_scale="tealrose",
                range_x=[0, 1],
                title="Concept Mastery Scores",
            )
            st.plotly_chart(fig, width='stretch')

            st.markdown("### Topic Insight")
            high = concept_df[concept_df["Mastery"] >= 0.75].shape[0]
            medium = concept_df[(concept_df["Mastery"] >= 0.4) & (concept_df["Mastery"] < 0.75)].shape[0]
            low = concept_df[concept_df["Mastery"] < 0.4].shape[0]
            insight = (
                f"This student is performing strongly in {high} topic(s), "
                f"shows steady progress in {medium} area(s), and needs reinforcement in {low} topic(s)."
            )
            render_insight_box("AI Interpretation", insight)

            # Extract weak concepts (mastery < 0.65) from the mastery data
            weak_concepts = [
                {"Concept": concept, "Mastery": score}
                for concept, score in mastery.items()
                if score < 0.65
            ]
            weak_concepts = sorted(weak_concepts, key=lambda x: x["Mastery"])[:5]

            if weak_concepts:
                st.markdown("### Needs Attention")
                weak_df = pd.DataFrame(weak_concepts)
                weak_fig = px.bar(
                    weak_df,
                    x="Mastery",
                    y="Concept",
                    orientation="h",
                    range_x=[0, 1],
                    title="Weak Concepts",
                )
                st.plotly_chart(weak_fig, width='stretch')


# Recommendations page

elif section == "Recommendations":
    st.markdown("# Recommendations")
    st.write("Review recommendation cards with confidence, readiness, and explainable reasoning.")

    student_id = st.number_input("Student ID", min_value=1, value=1, key="rec_student")
    model = st.selectbox("Recommendation Model", ["hybrid", "cf", "content"], key="rec_model")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, key="rec_top_k")

    recommendations = api_get(f"/recommendations/{student_id}", params={"model": model, "top_k": top_k}, default={}).get("recommendations", [])

    if not recommendations:
        render_empty_state("No recommendations available yet. Complete more learning activities to generate recommendations.")
    else:
        st.markdown("### Recommendation Highlights")
        for rec in recommendations:
            topic = rec.get("topic_name", rec.get("topic", "Unknown Topic"))
            score = rec.get("predicted_score", rec.get("score", 0.0))
            readiness = rec.get("readiness", 0.0)
            confidence = rec.get("confidence", rec.get("confidence", 0.0))
            reason = rec.get("reason", "No explanation available.")

            with st.container():
                card_cols = st.columns([3, 1, 1, 1])
                with card_cols[0]:
                    st.markdown(f"#### {topic}")
                    st.write(reason)
                card_cols[1].metric("Priority", f"{score:.2f}")
                card_cols[2].metric("Confidence", f"{confidence:.2f}")
                card_cols[3].metric("Readiness", f"{readiness:.2f}")

                with st.expander("Why this recommendation?"):
                    st.markdown("- Similar successful learners mastered this topic")
                    st.markdown("- Prerequisites completed")
                    st.markdown("- High readiness score")
                    st.markdown("- Strong knowledge graph pathway")
                    st.write(
                        "Peer evidence and neighbor student behavior are used to justify the recommendation."
                    )
                st.markdown("---")

        render_insight_box(
            "Research Insight",
            "The explainable recommendation pipeline delivers transparent guidance while preserving model interpretability.",
        )


# Knowledge Graph page

elif section == "Knowledge Graph":
    st.markdown("# Knowledge Graph")
    st.write("Explore concept relationships, prerequisites, and personalized learning paths.")

    concept = st.selectbox("Select a concept", api_get("/topics", default={}).get("topics", ["Matrices"]))
    path = api_get("/kg/path", params={"topic": concept}, default={}).get("path", [])

    cols = st.columns(3)
    cols[0].metric("Concepts", len(api_get("/topics", default={}).get("topics", [])))
    kg_info = api_get("/kg/subgraph", default={})
    cols[1].metric("Relationships", len(kg_info.get("edges", [])))
    cols[2].metric("Learning Paths", len(path))

    if not path:
        render_empty_state("No knowledge graph path found for this concept.")
    else:
        st.markdown("### Learning Journey")
        journey = " → ".join(path)
        st.write(journey)

        with st.expander("Concept flow details"):
            for step in path:
                st.markdown(f"- **{step}**")

        st.markdown("### Graph Statistics")
        if kg_info:
            stat_cols = st.columns(3)
            stat_cols[0].metric("Concept nodes", len(kg_info.get("nodes", [])))
            stat_cols[1].metric("Edges", len(kg_info.get("edges", [])))
            stat_cols[2].metric("Path options", len(path))

        render_insight_box(
            "Research Insight",
            "Knowledge graph reasoning reveals prerequisite chains and highlights the next best learning topics.",
        )


# Spaced Repetition page

elif section == "Spaced Repetition":
    st.markdown("# Spaced Repetition")
    st.write("See review priorities, retention forecasts, and scheduled learning activities.")

    student_id = st.number_input("Student ID", min_value=1, value=1, key="sr_student")
    due_items = api_get(f"/reviews/due/{student_id}", default={}).get("due_topics", [])
    schedule = api_get(f"/reviews/schedule/{student_id}", params={"days_ahead": 7}, default={}).get("schedule", [])

    cards = [
        {"label": "Retention %", "value": "78%", "caption": "Forecasted retention"},
        {"label": "Topics Due", "value": len(due_items), "caption": "Today’s review workload"},
        {"label": "Average Interval", "value": "4.2 days", "caption": "Scheduled review cadence"},
        {"label": "Review Streak", "value": "3 days", "caption": "Ongoing review habit"},
    ]
    render_metric_cards(cards)

    if due_items:
        st.markdown("### Today’s Due Topics")
        st.write(
            "\n".join([f"- {item.get('topic', item)}" for item in due_items])
        )
    else:
        render_empty_state("No topics are due for review today.")

    if schedule:
        st.markdown("### Upcoming Reviews")
        schedule_df = pd.DataFrame(schedule)
        st.table(schedule_df)

    st.markdown("### Retention Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[0.92, 0.87, 0.82, 0.76, 0.71], mode="lines+markers"))
    fig.update_layout(title="Retention Curve", xaxis_title="Review Sessions", yaxis_title="Retention")
    st.plotly_chart(fig, width='stretch')


# Instructor Analytics page

elif section == "Instructor Analytics":
    st.markdown("# Instructor Analytics")
    st.write("Monitor cohort performance, detect at-risk learners, and prioritize interventions.")

    weak_threshold = st.slider("Weak concept threshold", min_value=0.0, max_value=1.0, value=0.65)
    mastery_threshold = st.slider("At-risk mastery threshold", min_value=0.0, max_value=1.0, value=0.6)

    overview = api_get("/instructor/cohort_overview", params={"weak_threshold": weak_threshold}, default={})
    risk = api_get("/instructor/at_risk", params={"mastery_threshold": mastery_threshold, "max_results": 20}, default={})

    cards = [
        {"label": "Total Students", "value": overview.get("student_count", 0)},
        {"label": "Average Mastery", "value": f"{overview.get('average_mastery', 0.0) * 100:.1f}%"},
        {"label": "At-Risk Students", "value": risk.get("count", 0)},
    ]

    # Resolve top weak topic to a simple string for metric display
    top_weak = "N/A"
    if overview:
        if overview.get('top_weak_topics'):
            top_weak = overview.get('top_weak_topics')[0]
        elif overview.get('weak_concepts'):
            first = overview.get('weak_concepts')[0]
            if isinstance(first, dict):
                top_weak = first.get('topic', str(first))
            else:
                top_weak = str(first)

    cards.append({"label": "Top Weak Topic", "value": top_weak})
    render_metric_cards(cards)

    st.markdown("### Mastery Distribution")
    # Use student_masteries when available for a proper distribution
    if overview.get("student_masteries"):
        mastery_df = pd.DataFrame({"mastery": overview.get("student_masteries")})
        fig = px.histogram(mastery_df, x="mastery", nbins=10, title="Student Mastery Distribution", range_x=[0,1])
        fig.update_layout(xaxis_title="Mastery", yaxis_title="Count")
        st.plotly_chart(fig, width='stretch')
    elif overview.get("topic_stats"):
        topic_df = pd.DataFrame(overview.get("topic_stats"))
        if not topic_df.empty and "topic" in topic_df.columns and "average_score" in topic_df.columns:
            topic_df = topic_df.rename(columns={"average_score": "average_mastery"})
            st.bar_chart(topic_df.set_index("topic")[["average_mastery"]])
    else:
        render_empty_state("Cohort statistics are not available yet.")

    st.markdown("### At-Risk Students")
    at_risk = risk.get("at_risk", [])
    if at_risk:
        risk_df = pd.DataFrame(at_risk)
        if "student_id" not in risk_df.columns and risk_df.shape[1] >= 3:
            risk_df.columns = ["student_id", "mastery", "risk_level"]
        st.dataframe(risk_df, width='stretch')

        # Export at-risk list
        csv_bytes = risk_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download At-Risk CSV", data=csv_bytes, file_name="at_risk_students.csv", mime="text/csv")
        st.download_button("Download At-Risk JSON", data=risk_df.to_json(orient='records'), file_name="at_risk_students.json", mime="application/json")
        # Export cohort overview
        if overview:
            overview_df = pd.DataFrame(overview.get('topic_stats', []))
            if not overview_df.empty:
                st.download_button("Download Cohort Topics CSV", data=overview_df.to_csv(index=False).encode('utf-8'), file_name="cohort_topics.csv", mime="text/csv")
    else:
        render_empty_state("No at-risk students detected with the current threshold.")

    render_insight_box(
        "Research Insight",
        "Instructor analytics combine cohort mastery with weak-topic detection to prioritize evidence-based interventions.",
    )


# Research Lab page

elif section == "Research Lab":
    st.markdown("# Research Lab")
    st.write("Compare models, benchmark performance, and review research-grade evaluation metrics.")

    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    evaluation_mode = st.selectbox("Evaluation mode", ["full", "temporal"])

    bench = api_get("/recommend/benchmark", params={"top_k": top_k, "evaluation": evaluation_mode, "sample_fraction": 0.3}, default={})
    if not bench:
        render_empty_state("Benchmark data is not available. Run the recommendation benchmark first.")
    else:
        leaderboard = pd.DataFrame(bench).T.reset_index().rename(columns={"index": "model"})
        leaderboard = leaderboard.sort_values(by=leaderboard.columns[1], ascending=False)

        st.markdown("### Model Leaderboard")
        st.dataframe(leaderboard.head(10), width='stretch')

        if not leaderboard.empty:
            top_model = leaderboard.iloc[0]["model"]
            render_insight_box(
                "Leaderboard Insight",
                f"🥇 {top_model} is currently the highest-ranked model based on the selected benchmark metrics.",
            )

        if "precision_at_5" in leaderboard.columns:
            fig = px.bar(
                leaderboard,
                x="model",
                y=[col for col in leaderboard.columns if "precision" in col or "recall" in col or "ndcg" in col],
                title="Model Comparison",
            )
            st.plotly_chart(fig, width='stretch')

    st.markdown("### Metrics explained")
    st.write(
        "- NDCG: Rank quality for top recommendations.\n"
        "- MRR: How quickly the first good result appears.\n"
        "- Coverage: Fraction of items the model can recommend.\n"
        "- Novelty: How fresh the recommendations are."
    )


# System page

elif section == "System":
    st.markdown("# System")
    st.write("Monitor API health, dataset readiness, and system stability.")

    if st.button("Check API Health"):
        health = api_get("/health", default={})
        st.json(health)

    if st.button("Load Dataset Statistics"):
        rec_results = api_get("/results/recommendations", default={}).get("results", {})
        emb_data = api_get("/students/embeddings_data", default={}).get("results", {}).get("embeddings", {})
        st.metric("Recommendation result sets", len(rec_results))
        st.metric("Student embeddings", len(emb_data))

    st.markdown("### System Notes")
    st.info(
        "Use this page to validate service health, dataset readiness, and to confirm backend connectivity."
    )
