import os 
import re 
import json 
import uuid 
from pathlib import Path 
from datetime import datetime, timezone 
 
SESSIONS_DIR = Path("sessions") 
 
def migrate_txt_to_json(): 
    migrated = 0 
    for txt_file in SESSIONS_DIR.glob("*.txt"): 
        try: 
            # Skip empty files
            if txt_file.stat().st_size == 0:
                print(f'Skipped empty file: {txt_file.name}')
                continue
                
            with open(txt_file, 'r', encoding='utf-8') as f: 
                content = f.read() 
            
            messages = [] 
            for line in content.strip().split('\n'): 
                # Fixed regex - the issue was [\]] should be [^\]]
                match = re.match(r'\[([^\]]+)\]\s+(user|assistant):\s+(.+)', line, re.DOTALL) 
                if match: 
                    ts, role, msg = match.groups() 
                    messages.append({
                        'role': role, 
                        'content': msg.strip(), 
                        'timestamp': ts
                    }) 
            
            if messages: 
                # Get first user message for title
                first_msg = next((m for m in messages if m['role'] == 'user'), None) 
                if first_msg:
                    # Create a meaningful title from first message
                    title = ' '.join(first_msg['content'].split()[:5]).capitalize()
                    # Clean up title
                    title = title.replace('"', '').replace("'", "")[:50]
                else:
                    title = 'Imported Chat'
                
                sid = str(uuid.uuid4()) 
                data = {
                    'session_id': sid, 
                    'title': title, 
                    'created_at': datetime.now(timezone.utc).isoformat(), 
                    'messages': messages
                } 
                json_path = SESSIONS_DIR / f'session_{sid}_{txt_file.stem}.json' 
                
                with open(json_path, 'w', encoding='utf-8') as f: 
                    json.dump(data, f, indent=2, ensure_ascii=False) 
                
                print(f'✓ Migrated: {txt_file.name} -> {json_path.name}')
                print(f'  Title: "{title}"')
                print(f'  Messages: {len(messages)}')
                migrated += 1 
            else:
                print(f'✗ No valid messages found in: {txt_file.name}')
                
        except Exception as e: 
            print(f'✗ Error migrating {txt_file.name}: {e}') 
    
    print(f'\n{"="*50}')
    print(f'Migration Complete!')
    print(f'{"="*50}')
    print(f'Successfully migrated: {migrated} sessions') 
 
if __name__ == '__main__': 
    print('='*50)
    print('Starting Migration of .txt sessions to .json format')
    print('='*50)
    print()
    migrate_txt_to_json()
