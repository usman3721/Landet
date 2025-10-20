from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Language Detector API is running!"}
