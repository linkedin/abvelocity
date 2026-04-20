
cd $HOME/codes/my_repos/abvelocity/docs/static/test-results/mea/mea-guides/
if ! command -v node &> /dev/null
then
  echo "Error: node.js is not installed. Please install it."
  echo "You can use Homebrew: 'brew install node'"
  exit 1
fi
if [ ! -d "node_modules/puppeteer" ]; then
  echo "Puppeteer is not installed. Installing..."
  npm install puppeteer --silent
  npx puppeteer browsers install chrome --silent
  sleep 5
fi
BASE_DIR="$HOME/codes/my_repos/abvelocity/docs/static/test-results/mea/mea-guides/"
FILE_BASE_NAMES=(
  "mea-report"
  "mea-report-scenario"
  "mea-report-low-overlap"
)
for BASE_NAME in "${FILE_BASE_NAMES[@]}"; do
  INPUT_FILE="${BASE_DIR}${BASE_NAME}.html"
  OUTPUT_FILE="${BASE_DIR}${BASE_NAME}.pdf"
  if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found. Skipping."
    continue
  fi
  JS_SCRIPT=$(mktemp --tmpdir=. -t 'puppeteer-script.XXXXXXXXXX')
  cat > "$JS_SCRIPT" << EOF
const puppeteer = require('puppeteer');
(async () => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    const filePath = 'file://${INPUT_FILE}';
    await page.goto(filePath, {
        waitUntil: 'networkidle0'
    });
    await page.pdf({
        path: '${OUTPUT_FILE}',
        format: 'A4',
        printBackground: true
    });
    await browser.close();
    console.log('PDF generated successfully!');
})();
EOF
  node "$JS_SCRIPT"
  rm "$JS_SCRIPT"
  echo "Conversion complete for '$INPUT_FILE'."
done
echo "All specified files have been processed."
