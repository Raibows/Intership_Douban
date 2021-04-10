from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from predict import init, predict_movie_genres

app = FastAPI()
# 处理跨域问题
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, data_base, device = init()

# import rec_based_on_bert.predict as predict


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/movie/{name}")
async def search_movie_by_name(name: str = None):
    if name is None:
        return {"code": 404, "data": None}
    data = data_base.name_dict[name][1]
    return {"code": 200, "data": data}


@app.get("/type")
async def get_all_types():
    return {"code": 200, "data": data_base.label_uniq}


@app.get("/probability")
async def get_prob(name: str = None):
    if name is None:
        return {"code": 404, "data": None}
    logits, _ = predict_movie_genres(model, data_base, device, name)
    return {'coda': 200, 'data': logits}


if __name__ == '__main__':
    uvicorn.run(app='api:app', host="127.0.0.1", port=8000, reload=True, debug=True)
