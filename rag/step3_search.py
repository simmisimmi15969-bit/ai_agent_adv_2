from pathlib import Path
import chromadb 
from step2_embed import embed
DB_DIR = Path(__file__).parent / "chroma_db"
client = chromadb.PersistentClient(path=str(DB_DIR))
collection= client.get_collection("my_docs")

def search(question:str, n_result:int=3) -> list[dict]:
    question_vector = embed([question])[0]
    result = collection.query(
        query_embeddings=[question_vector],
        n_results=n_result,
        include=["documents","metadatas", "distances"],
    )
    matches=[]
    for i in range(len(result["ids"][0])):
     matches.append({
       "id": result["ids"][0][i],
       "text": result["documents"][0][i],
       "source": result["metadatas"][0][i]["source"],
       "distance": result["distances"][0][i],    
       "page": result["metadatas"][0][i].get("page"),
   
     })

    return matches

if __name__ =="__main__":
    question="Can i work from my house?"
    print(f"question: {question}\n")

    for rank, match in enumerate(search(question), start=1):
        print(f"#{rank}  {match['id']}  (distance {match['distance']:.3f}, similarity {1 - match['distance']:.3f})")
        print(f"    source: {match['source']}")
        print(f"    text:   {match['text'].strip()[:120]}...\n")
