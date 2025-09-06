import csv
with open("../datasets/cai-vision-dataset/classes.csv", newline="", encoding="utf-8") as f, open("labels.txt","w",encoding="utf-8") as out:
    rdr = csv.DictReader(f)
    rows = sorted(rdr, key=lambda r: int(r["class_id"]))
    for r in rows:
        out.write(f'{r["display_name_en"]}\n')
