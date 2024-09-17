from fastapi import FastAPI
from router import image_routes

app = FastAPI()

# Include the router from the router/image_routes.py
app.include_router(image_routes.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
