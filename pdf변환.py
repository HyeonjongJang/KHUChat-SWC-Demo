import os
import win32com.client

source_folder = r"C:\Users\aims0\Desktop\2025_1\chatbot data docx"
target_folder = r"C:\Users\aims0\Desktop\2025_1\chatbot data pdf"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

word = win32com.client.Dispatch("Word.Application")
word.Visible = False

for filename in os.listdir(source_folder):
    if filename.endswith(".docx"):
        doc_path = os.path.join(source_folder, filename)
        pdf_path = os.path.join(target_folder, filename.replace(".docx", ".pdf"))
        doc = word.Documents.Open(doc_path)
        doc.SaveAs(pdf_path, FileFormat=17)  # 17 = PDF
        doc.Close()
word.Quit()
