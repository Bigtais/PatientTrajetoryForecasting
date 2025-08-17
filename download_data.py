import os
from umls_downloader import download_umls_full

# Get this from https://uts.nlm.nih.gov/uts/edit-profile
api_key = "7623b78e-6071-4346-b202-745882c637cd"

path = download_umls_full(version="2025AA", api_key=api_key)

# This is where it gets downloaded: ~/.data/bio/umls/2025AA/umls-2025AA-mrconso.zip