import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_mining.competitor_spider import CompetitorSpider

def debug_generation():
    spider = CompetitorSpider()
    term = "monoblocchi termoisolanti"
    print(f"DEBUG: Generating for term: {term}")
    
    prompt = f"""[INST]
    Write a technical sentence in Italian containing the term: "{term}".
    Then translate it into English, French, and Spanish.
    
    Format:
    IT: [Sentence]
    EN: [Translation]
    FR: [Translation]
    ES: [Translation]
    [/INST]"""
    
    out = spider.ai.pipe(prompt, max_new_tokens=200)[0]['generated_text']
    print("-" * 20)
    print("RAW OUTPUT:")
    print(out)
    print("-" * 20)
    
    parts = out.split("[/INST]")[-1].strip().split('\n')
    print("PARTS AFTER SPLIT:")
    for i, p in enumerate(parts):
        print(f"Part {i}: '{p}'")
    
    row = {"term_keyword": term, "source_lang": "ita_Latn", "origin": "web_spider"}
    found_it = False
    found_en = False
    for line in parts:
        if line.startswith("IT:"): 
            row["source_text"] = line.replace("IT:", "").strip()
            found_it = True
        if line.startswith("EN:"): 
            row["target_text_en"] = line.replace("EN:", "").strip()
            found_en = True
            
    print("-" * 20)
    print(f"Found IT: {found_it}")
    print(f"Found EN: {found_en}")
    print(f"Final Row: {row}")

if __name__ == "__main__":
    debug_generation()
