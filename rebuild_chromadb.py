import os  
import json  
from pathlib import Path  
import sys  
  
# Add backend to path  
sys.path.insert(0, str(Path(__file__).parent / "backend"))  
from server import ChromaMemoryStore, SESSIONS_DIR  
  
print("Rebuilding ChromaDB from existing sessions...")  
store = ChromaMemoryStore()  
  
count = 0  
for json_file in SESSIONS_DIR.glob("*.json"):  
    try:  
        with open(json_file, 'r', encoding='utf-8') as f:  
            data = json.load(f)  
        for msg in data.get("messages", []):  
            role = msg.get("role")  
            content = msg.get("content")  
            if role and content:  
                store.add_text(f"{role}: {content}", source_file=json_file, metadata={"role": role, "session_id": data.get("session_id")})  
                count += 1  
        print(f"Processed: {json_file.name} - {len(data.get('messages', []))} messages")  
    except Exception as e:  
        print(f"Error processing {json_file.name}: {e}")  
  
print(f"\nDone! Rebuilt ChromaDB with {count} messages from {len(list(SESSIONS_DIR.glob('*.json')))} sessions") 
