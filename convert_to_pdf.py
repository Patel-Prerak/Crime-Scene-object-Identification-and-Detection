import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md, output_pdf):
    # 1. Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Convert to HTML
    html_content = markdown.markdown(text)
    
    # Fix image paths to be absolute
    current_dir = os.getcwd().replace('\\', '/')
    html_content = html_content.replace('src="./', f'src="{current_dir}/')

    # 3. Add Styling for PDF
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: Helvetica, sans-serif;
                font-size: 11pt;
                line-height: 1.5;
            }}
            h1 {{
                font-size: 18pt;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 20px;
            }}
            h2 {{
                font-size: 14pt;
                color: #34495e;
                margin-top: 15px;
                border-bottom: 1px solid #ddd;
            }}
            h3 {{
                font-size: 12pt;
                color: #555;
                font-weight: bold;
            }}
            p {{
                text-align: justify;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px auto;
                display: block;
            }}
            .caption {{
                font-size: 9pt;
                color: #666;
                text-align: center;
                font-style: italic;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # 4. Save to PDF
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            full_html,
            dest=result_file,
            encoding='utf-8'
        )

    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
    else:
        print(f"Successfully created PDF: {output_pdf}")

if __name__ == "__main__":
    convert_md_to_pdf("IEEE_Project_Report_NFSU.md", "IEEE_Project_Report_NFSU.pdf")
