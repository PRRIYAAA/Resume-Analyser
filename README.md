# AI Resume Analyzer & ATS Optimizer

A powerful AI-powered tool to analyze resumes, provide feedback, and optimize for ATS (Applicant Tracking Systems) using machine learning and OpenAI's GPT models.

## 🚀 Features

- **AI Resume Scoring & Feedback**: Get detailed, actionable feedback on your resume from an AI HR interviewer perspective
- **ML-based ATS Matching**: Uses TF-IDF vectorization to calculate ATS compatibility scores
- **Smart Rewrite Suggestions**: AI-generated suggestions to improve resume sections for better ATS performance
- **Multi-format Support**: Supports PDF and DOCX resume formats
- **Fresher-Friendly**: Designed specifically for entry-level candidates

## 📋 Requirements

- Python 3.8+
- OpenAI API Key
- Internet connection for AI analysis

## 🛠️ Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd Resume-Analyser
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## 🎯 Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The app will open at `http://localhost:8501`

3. **Upload and analyze**
   - Upload your resume (PDF or DOCX)
   - Optionally paste a job description for ATS matching
   - Click "Analyze Resume" to get AI feedback and ATS insights

## 📊 How It Works

### AI Resume Analysis
- Uses OpenAI's GPT-4o-mini model to evaluate resumes
- Provides structured feedback including:
  - Resume score out of 100
  - Strengths identification
  - Detailed improvement suggestions
  - Role readiness assessment

### ATS Optimization
- Employs TF-IDF (Term Frequency-Inverse Document Frequency) algorithm
- Compares resume content against job description keywords
- Identifies matched and missing keywords
- Generates rewrite suggestions for better ATS compatibility

### Supported File Formats
- **PDF**: Uses PyPDF2 for text extraction
- **DOCX**: Uses python-docx for document parsing

## 🔧 Dependencies

- `streamlit` - Web app framework
- `PyPDF2` - PDF text extraction
- `python-docx` - DOCX text extraction
- `openai` - AI model integration
- `scikit-learn` - Machine learning algorithms
- `python-dotenv` - Environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open-source. Please check the license file for details.

## ⚠️ Disclaimer

This tool is designed to assist with resume optimization but does not guarantee job placement. Always review AI suggestions critically and tailor them to your specific situation.

## 🆘 Troubleshooting

- **API Key Issues**: Ensure your OpenAI API key is valid and has sufficient credits
- **File Upload Errors**: Check that your resume is not password-protected and is in supported format
- **Installation Problems**: Make sure you have Python 3.8+ installed

---

Built with ❤️ using Streamlit, OpenAI, and Scikit-learn