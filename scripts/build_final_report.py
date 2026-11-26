from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'reports/source/report_content.json').read_text())
def setup(d,chinese=False):
 sec=d.sections[0]; sec.top_margin=Inches(.67);sec.bottom_margin=Inches(.62);sec.left_margin=sec.right_margin=Inches(.72)
 sec.page_width=Inches(8.5); sec.page_height=Inches(11)
 styles=d.styles
 for name in ['Normal','Title','Heading 1','Heading 2','Caption']:
  st=styles[name];st.font.name='Arial Unicode MS' if chinese else 'Arial';st.font.color.rgb=RGBColor(0,0,0)
  st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Arial Unicode MS')
 for st in styles:
  for border in st._element.xpath('.//w:pBdr'): border.getparent().remove(border)
 styles['Normal'].font.size=Pt(10.5 if not chinese else 11)
 styles['Normal'].paragraph_format.space_after=Pt(7)
 styles['Normal'].paragraph_format.line_spacing=1.08
 styles['Heading 1'].font.size=Pt(20);styles['Heading 2'].font.size=Pt(12)
 styles['Caption'].font.size=Pt(9);styles['Caption'].font.italic=False
 styles['Heading 1'].paragraph_format.space_after=Pt(12)
 foot=sec.footer.paragraphs[0];foot.alignment=2
 run=foot.add_run('BIOSTAT 707  •  ');run.font.size=Pt(8)
 fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)
def addtable(d,matrix):
 t=d.add_table(rows=1, cols=len(matrix[0]));t.autofit=False
 width=7.06/len(matrix[0]);
 if len(matrix[0])==2: widths=[1.6,5.46]
 else: widths=[width]*len(matrix[0])
 for col,w in zip(t.columns,widths):col.width=Inches(w)
 for i,row in enumerate(matrix):
  cells=t.rows[0].cells if i==0 else t.add_row().cells
  for c,txt,w in zip(cells,row,widths):
   c.width=Inches(w);c.text=str(txt)
   for pa in c.paragraphs:
    pa.paragraph_format.space_after=Pt(4);pa.paragraph_format.space_before=Pt(4)
    for r in pa.runs:r.font.size=Pt(9.5);r.bold=(i==0)
   tcpr=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D6D6D6');b.append(el)
   tcpr.append(b)
   if i==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EDEFF2');tcpr.append(sh)
  trpr=t.rows[i]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
 t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
 return t

d=Document();setup(d)
for i,pg in enumerate(report):
 if i:d.add_page_break()
 d.add_heading(pg['title'],0 if i==0 else 1)
 for b in pg['blocks']:
  if 'p'in b:d.add_paragraph(b['p'])
  elif 'h'in b:d.add_heading(b['h'],2)
  elif 'img'in b:d.add_picture(str(ROOT/b['img']),width=Inches(7.0));d.add_paragraph(b['caption'],'Caption')
  else:addtable(d,b['table']);d.add_paragraph(b['caption'],'Caption')
d.save(ROOT/'reports/final-report.docx')
