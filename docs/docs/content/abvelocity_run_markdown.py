# This is a Python script to generate the results based off the markdown file
# These results are stored then in the Library
# Author: Reza Hosseini
import subprocess
import sys
import os
import time


def install_package(package_name):
    """
    Installs a package using pip if it's not already installed.
    """
    try:
        __import__(package_name)
    except ImportError:
        print(f"{package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Successfully installed {package_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package_name}: {e}")
    else:
        print(f"{package_name} is already installed.")


def html_to_pdf(html_file, pdf_file):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Load your local HTML file
        page.goto(f"file://{os.path.abspath(html_file)}")

        # 'print_background=True' is the secret for table colors
        page.pdf(
            path=pdf_file,
            format="A4",
            print_background=True,
            margin={"top": "20px", "bottom": "20px", "left": "20px", "right": "20px"}
        )
        browser.close()


def run_markdown_code(file_name):
    install_package("markdown")
    install_package("beautifulsoup4")
    install_package("pyspellchecker")
    install_package("kaleido")
    install_package("playwright")
    import markdown
    from bs4 import BeautifulSoup

    file_path = os.path.join(os.path.expanduser("~"), file_name)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

    html = markdown.markdown(md_content, extensions=['fenced_code'])
    soup = BeautifulSoup(html, 'html.parser')
    code_blocks = [block.get_text() for block in soup.find_all('code', class_='language-python')]

    return code_blocks

home = os.path.expanduser("~")
fn = f"{home}/codes/my_repos/abvelocity/docs/docs/content/4-run-mea-generic.md"
code_blocks = run_markdown_code(fn)

for i, block in enumerate(code_blocks):
    print(f"\n\n\n\n\n***Running code block {i}...\n")
    print(block)
    exec(block)
    if i < len(code_blocks) - 1:
        print(f"Pausing for 1 seconds before the next block...\n")
        time.sleep(1)

# Create pdf versions of the reports
paths = [
    f"{home}/codes/my_repos/abvelocity/docs/static/test-results/mea/mea-guides",
    # f"{home}/codes/rzdv/mea",
    # f"{home}/codes/my_repos/abvelocity/docs/docs/content",
]

files = [
    "mea-report",
    "mea-report-low-overlap",
    "mea-report-scenario",
]

for path in paths:
    for file in files:
        html_file = f"{path}/{file}.html"
        pdf_file = f"{path}/{file}.pdf"
        html_to_pdf(html_file, pdf_file)
