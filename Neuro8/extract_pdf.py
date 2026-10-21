from pypdf import PdfReader
r = PdfReader(r".zwork/attachments/1e783906-cdec-436e-90b0-daa9c1bc8145/NEUROWEAVE_SPIKER_единый_концепт_v4.0.pdf")
print("PAGES", len(r.pages))
out=[]
for i,p in enumerate(r.pages):
    t = p.extract_text() or ""
    out.append(f"\n===== PAGE {i+1} =====\n"+t)
open("concept_v4.txt","w",encoding="utf-8").write("".join(out))
print("CHARS", sum(len(x) for x in out))
