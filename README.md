# CodeSense AI - Python Edition

AI-Powered Code Analysis Platform built with Python and FastAPI.

## Project Overview

CodeSense AI is a web application that analyzes code using Google's Gemini AI. It provides instant feedback on code quality, error detection, optimization suggestions, and expected output for multiple programming languages.

## Features

- 🤖 **AI-Powered Analysis**: Uses Google Gemini AI for comprehensive code analysis
- 🌐 **Multi-Language Support**: Supports JavaScript, Python, Java, C++, C#, TypeScript, Go, Rust, and more
- ⚡ **Real-time Analysis**: Get instant feedback on code quality, errors, and optimizations
- 🔒 **Security Focused**: Identifies security vulnerabilities and best practices
- 🎨 **Modern UI**: Clean, responsive interface built with vanilla HTML/CSS/JavaScript

## Technologies Used

- **Backend**: FastAPI (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **AI**: Google Gemini API
- **Server**: Uvicorn (ASGI server)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd codesense-ai-powered-bb1a507a1a0dcc7473ac740b35e2bca202fa1d83
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env  # Windows
     # or
     cp .env.example .env    # macOS/Linux
     ```
   - Edit `.env` and add your Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

## Running the Application

1. **Start the development server**:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

## Project Structure

```
.
├── main.py                 # Application entry point
├── app.py                  # API routes and logic
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .env                   # Your environment variables (create this)
├── templates/
│   └── index.html         # Main HTML page
├── static/
│   ├── css/
│   │   └── style.css      # Stylesheet
│   └── js/
│       └── app.js         # Frontend JavaScript
└── README.md              # This file
```

## API Endpoints

### POST `/api/analyze`

Analyzes code using AI.

**Request Body:**
```json
{
  "code": "print('Hello, World!')",
  "language": "python"
}
```

**Response:**
```json
{
  "errors": [
    {
      "line": 1,
      "message": "Error description",
      "severity": "error"
    }
  ],
  "suggestions": ["Suggestion 1", "Suggestion 2"],
  "optimizations": ["Optimization 1", "Optimization 2"],
  "output": "Expected output"
}
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `PORT`: Server port (default: 8000)

## Development

### Running in Development Mode

The application runs with auto-reload enabled by default. Any changes to Python files will automatically restart the server.

### Making Changes

- **Backend**: Edit `app.py` for API logic, `main.py` for app configuration
- **Frontend**: Edit files in `templates/` and `static/` directories

## Deployment

### Production Deployment

For production, use a production ASGI server like Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Or deploy to platforms like:
- **Heroku**: Add a `Procfile` with `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Railway**: Configure start command
- **DigitalOcean App Platform**: Set start command
- **AWS/GCP/Azure**: Use containerized deployment

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY is not configured"**
   - Make sure you've created a `.env` file with your API key
   - Verify the key is correct and has no extra spaces

2. **Port already in use**
   - Change the `PORT` in `.env` or use a different port
   - Kill the process using the port

3. **Module not found errors**
   - Make sure you've activated your virtual environment
   - Run `pip install -r requirements.txt` again

## License

This project is open source and available for use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the repository.
