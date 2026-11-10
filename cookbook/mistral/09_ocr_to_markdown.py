import sys

from client import client

if len(sys.argv) < 2:
    print("사용법: python 09_ocr_to_markdown.py <파일.pdf|이미지> [출력.md]")
    sys.exit(1)

path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else "output.md"

with open(path, "rb") as f:
    uploaded = client.files.upload(file={"file_name": path.split("/")[-1], "content": f})

signed = client.files.get_signed_url(file_id=uploaded.id)
result = client.ocr.process(
    model="mistral-ocr-latest",
    document={"type": "document_url", "document_url": signed.url},
)

markdown = "\n\n---\n\n".join(page.markdown for page in result.pages)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(markdown)

print(f"{len(result.pages)}페이지를 {out_path}에 저장했습니다.")
