import json
import os
from util import load_index_in_memory
from booleanRetrieval import booleanRetrieval
from init import index
from fastapi import FastAPI,Request,HTTPException,Response
import uvicorn
import redis


r = redis.StrictRedis(
  host='127.0.0.1',
  port=6379,
  decode_responses=True
  )
f = open("s2/s2_doc.json", encoding="utf-8")
json_file = json.load(f)

#initialize server
app = FastAPI()

async def isPostingCreated():
    resp =  r.exists('doc_freq')
    return resp

@app.post('/boolean-retreival')
async def getBooleanDocs(req:Request):
   try:
      data = await req.json()
      if "query" not in data:
            raise HTTPException(
                status_code=422, detail="Incomplete data provided")
      query = data['query']
      return Response(content=booleanRetrieval(query,json_file,r),media_type="application/json")
   except Exception as e:
      raise e
   



if __name__ == "__main__":
   
   if(r.exists('doc_freq')!=1):
      print('Creating Postings')
      index('s2/',redis=r)

   # postings,doc_freq = load_index_in_memory('s2/')

   uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)




