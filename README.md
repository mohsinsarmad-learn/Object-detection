# Project Name

Object Detection with yolov modelS

## Prerequisites

- Python 3.6 or higher
- Git (to clone the repository)

## Installation

1. Clone the repository

   ```bash
   https://github.com/mohsinsarmad-learn/Object-detection.git
   cd server
   ```

2. Create a virtual environment

   ```bash
   # Windows (PowerShell)
   python -m venv venv
   ```

3. Activate the virtual environment

   ```bash

   # Windows (PowerShell)
   .\venv\Scripts\Activate
   ```

4. (Optional) Upgrade pip

   ```bash
   pip install --upgrade pip
   ```

5. Install all required packages
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Once dependencies are installed run:

```bash
uvicorn app.main:app --reload
```

When you will run this command at the first time it will download model from the github

### Testing

You should have postman installed for next steps

1. Open psotman.
2. Create new POST request.

```
http://localhost:8000/detect-video/
```

3. Navigate to body and select "form-data".
4. key : Input->("file") | Type->("File")-Select from dropdown. | Value: Input->("select video you want to input").
5. key : Input->("conf") | Type->("Text")-Select from dropdown. | Value: Input->("0.5").
6. Click "Send".
7. Now wait for the mdel to do its work.
8. After some time if the ("status : 200") then move to next step.
9. Click on three dots on the right side of the status at the end of the line and "save response to file".
10. Nevigate to the video and open it to see the result.

### Deactivation

When you’re finished working, deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```
server/
├── venv/               # Virtual environment directory
├── .gitignore
├── README.md
├── requirements.txt
└── app/
    ├── __pycache__     # After running  script once
    ├── detection.py
    └── main.py         # Entry-point script
```

- **venv/** – your virtual environment (ignored by git)
- **.gitignore** – patterns to exclude from version control
- **requirements.txt** – pinned Python dependencies
- **app** – contain all the scripts
- **detection.py** – code related to model
- **main.py** - Entry-point script
