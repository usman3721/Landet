# 🌍 Language Detector

A scalable, production-ready **Language Detection System** built with **FastAPI**, **DVC**, and **MLOps best practices**.  
Designed for high-performance text classification and language prediction pipelines with monitoring, caching, and model tracking.

---

## 🧠 Features

- 🚀 **FastAPI Backend** – Clean RESTful API endpoints  
- 🔒 **JWT Authentication** – Secure token-based user access  
- ⚙️ **MLOps Integration** – DVC for data versioning, MLflow for experiment tracking  
- ⚡ **Redis Caching** – Fast model inference with caching  
- 📊 **Prometheus Monitoring** – System metrics for performance tracking  
- 🧩 **Modular Structure** – Scalable and extensible code architecture

---

## 🏗️ Project Structure


---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/language-detector.git
cd language-detector
```


## 2️⃣ Install Dependencies

Make sure you have Poetry installed:

```bash
pip install poetry
```
Then:
```bash
poetry install
```
3️⃣ Activate the Virtual Environment
```bash
language-detector/
│
├── app/
│   ├── api/
│   │   ├── routes_auth.py         # Authentication endpoints
│   │   ├── routes_predict.py      # Language prediction endpoint
│   │
│   ├── core/
│   │   ├── config.py              # Environment variables & settings
│   │   ├── dependencies.py        # API key & JWT dependencies
│   │   ├── exceptions.py          # Global exception handling
│   │   ├── security.py            # JWT token generation/verification
│   │
│   ├── middleware/
│   │   ├── logging_middleware.py  # Request/response logging
│   │
│   ├── services/
│   │   ├── language_service.py    # Model loading & prediction logic
│   │
│   ├── utils/
│   │   ├── redis_cache.py         # Redis caching functions
│   │
│   ├── models/
│   │   └── model.joblib           # Trained ML model file
│   │
│   └── main.py                    # FastAPI app entry point
│
├── data/                          # Dataset and versioned files (DVC)
├── mlruns/                        # MLflow experiment logs
├── dvc.yaml                       # DVC pipeline configuration
├── poetry.lock                    # Poetry lock file
├── pyproject.toml                 # Project dependencies and metadata
├── .env                           # Environment variables
└── README.md                      # Documentation
```


---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/usmanazan3721/language-detector.git
cd language-detector
