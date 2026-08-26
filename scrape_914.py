import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://en.scpslgame.com/index.php?title=SCP-914/Outputs"
soup = BeautifulSoup(requests.get(url).text, "html.parser")

cols = ["Input", "Rough", "Coarse", "1:1", "Fine", "Very Fine"]
tables = []

for table in soup.find_all("table"):
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    if headers[:6] == cols:
        prev = table.find_previous(["h2", "h3", "h4"])
        admin_only = prev and "Remote Admin Only" in prev.get_text()
        rows = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(separator=", ", strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells[:6] + [admin_only])
        tables.append(pd.DataFrame(rows, columns=cols + ["admin_only"]))

pd.concat(tables, ignore_index=True).to_csv("914_outputs.csv", index=False)

print("Saved 914_outputs.csv")
