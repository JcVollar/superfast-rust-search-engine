#!/usr/bin/env python3
"""
Convert Norwegian business registry CSV (Brønnøysundregistrene) to ProductDocument JSONL.

Field mapping:
  product_id   <- organisasjonsnummer (org number, unique ID)
  name         <- navn (business name)
  brand        <- organisasjonsform.beskrivelse (org type: Aksjeselskap, etc.)
  category     <- naeringskode1.beskrivelse (industry code description)
  sku          <- organisasjonsnummer
  color        <- forretningsadresse.poststed (city)
  size         <- forretningsadresse.postnummer (zip code)
  material     <- forretningsadresse.kommune (municipality)
  description  <- vedtektsfestetFormaal (registered purpose)
  specifications <- aktivitet (activity description)
  price        <- antallAnsatte (number of employees, as float)
  stock        <- 0
  attributes   <- extra fields (hjemmeside, epostadresse, telefon, etc.)

Usage:
  python csv_to_jsonl.py input.csv output.jsonl
"""

import csv
import json
import sys


def convert(input_path: str, output_path: str):
    count = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)

        for row in reader:
            org_nr = row.get("organisasjonsnummer", "").strip()
            if not org_nr:
                skipped += 1
                continue

            navn = row.get("navn", "").strip()
            if not navn:
                skipped += 1
                continue

            # Parse employees as number
            try:
                ansatte = int(row.get("antallAnsatte", "0").strip() or "0")
            except ValueError:
                ansatte = 0

            # Build attributes with extra useful fields
            attributes = {}
            for key in [
                "hjemmeside", "epostadresse", "telefon", "mobil",
                "organisasjonsform.kode", "naeringskode1.kode",
                "naeringskode2.beskrivelse", "naeringskode3.beskrivelse",
                "postadresse.adresse", "postadresse.poststed", "postadresse.postnummer",
                "forretningsadresse.adresse",
                "institusjonellSektorkode.beskrivelse",
                "stiftelsesdato", "registreringsdatoenhetsregisteret",
                "konkurs", "underAvvikling",
            ]:
                val = row.get(key, "").strip()
                if val:
                    attributes[key] = val

            # Build full address string for search (top-level TEXT field)
            addr_parts = []
            forr_addr = row.get("forretningsadresse.adresse", "").strip()
            if forr_addr:
                addr_parts.append(forr_addr)
            forr_post = row.get("forretningsadresse.poststed", "").strip()
            forr_zip = row.get("forretningsadresse.postnummer", "").strip()
            if forr_zip and forr_post:
                addr_parts.append(f"{forr_zip} {forr_post}")
            elif forr_post:
                addr_parts.append(forr_post)
            forr_kommune = row.get("forretningsadresse.kommune", "").strip()
            if forr_kommune and forr_kommune != forr_post:
                addr_parts.append(forr_kommune)
            full_address = ", ".join(addr_parts)

            # Also build a combined description with purpose + address
            purpose = row.get("vedtektsfestetFormaal", "").strip()
            description = purpose
            if full_address:
                description = f"{purpose} Adresse: {full_address}" if purpose else f"Adresse: {full_address}"

            doc = {
                "product_id": org_nr,
                "name": navn,
                "brand": row.get("organisasjonsform.beskrivelse", "").strip(),
                "category": row.get("naeringskode1.beskrivelse", "").strip(),
                "sku": org_nr,
                "color": forr_post,                                                 # city
                "size": forr_zip,                                                   # zip
                "material": forr_kommune,                                           # municipality
                "description": description,                                         # purpose + address
                "specifications": row.get("aktivitet", "").strip(),                 # activity
                "price": float(ansatte),
                "stock": 0,
                "attributes": attributes,
            }

            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

            if count % 100000 == 0:
                print(f"  Converted {count} records...", file=sys.stderr)

    print(f"Done: {count} records written, {skipped} skipped", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.csv> <output.jsonl>", file=sys.stderr)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
