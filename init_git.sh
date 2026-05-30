#!/bin/bash
# Elo Learn - Git Repository Initialization

echo "🚀 Initializing Elo Learn Git Repository..."

# Initialize git
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Elo Learn - Research-grade adaptive learning platform

Features:
- Recommendation systems (CF, content-based, hybrid)
- Reinforcement learning tutor with DQN
- Knowledge graph with concept dependencies
- Feature engineering pipeline
- Comprehensive API with FastAPI
- Docker containerization
- Extensive documentation and research findings

This is a production-grade platform demonstrating:
- Scalable ML systems architecture
- Personalized learning algorithms
- Educational AI research
- Best practices in system design"

# Create remote (if you have a GitHub repo URL, uncomment and update)
# git remote add origin https://github.com/your-username/elo-learn.git
# git branch -M main
# git push -u origin main

echo ""
echo "✅ Repository initialized!"
echo ""
echo "📝 Next steps:"
echo "  1. Add GitHub remote: git remote add origin <repo-url>"
echo "  2. Push to GitHub: git push -u origin main"
echo "  3. Update PROJECT_SUMMARY.md with your name and email"
echo ""
echo "📚 Documentation:"
echo "  - Start with: README.md"
echo "  - Setup: docs/QUICK_START.md"
echo "  - Architecture: docs/architecture.md"
echo "  - API: docs/api.md"
echo ""
echo "🚀 To get started immediately:"
echo "  python datasets/generate_synthetic_interactions.py"
echo "  python training/train_main.py"
echo ""
