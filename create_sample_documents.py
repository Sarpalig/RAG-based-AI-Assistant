from pathlib import Path
from docx import Document

base = Path('data/sample_documents')
base.mkdir(parents=True, exist_ok=True)

# Create DOCX with onboarding content
onboarding_doc = Document()
onboarding_doc.add_heading('Yeni Çalışan Onboarding Süreci', level=1)
onboarding_doc.add_paragraph('Yeni çalışanlar ilk gün şirket giriş prosedürlerini tamamlamalıdır.')
onboarding_doc.add_paragraph('İlk adım olarak, insan kaynakları ile tanışma toplantısı planlanır.')
onboarding_doc.add_paragraph('Daha sonra, IT ekibi tarafından bilgisayar ve erişim hakları hazırlanır.')
onboarding_doc.add_paragraph('Son olarak, çalışanlara şirket politikaları ve güvenlik kuralları anlatılır.')
onboarding_doc.save(base / 'onboarding_process.docx')

# Create a simple PDF-like text file for the sample set
pdf_content = "%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n4 0 obj\n<< /Length 144 >>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Yol Gideri Politikasi: Harcama talepleri ay sonundan once yoneticiye iletilmelidir.) Tj\nET\nendstream\nendobj\n5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\nxref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000062 00000 n \n0000000119 00000 n \n0000000207 00000 n \n0000000305 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
(base / 'travel_expense_policy.pdf').write_text(pdf_content, encoding='latin-1')
print('Created sample documents.')
