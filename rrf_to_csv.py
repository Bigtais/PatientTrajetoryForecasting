import pandas as pd

# Load MRREL.RRF with selected columns (pipe-delimited)
columns = [
    'CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2',
    'STYPE2', 'RELA', 'RUI', 'SRUI', 'SAB', 'SL',
    'RG', 'DIR', 'SUPPRESS', 'CVF'
]

df = pd.read_csv('UMLS/MRREL.RRF', sep='|', header=None, names=columns + ['extra'], dtype=str, usecols=['CUI1', 'REL', 'CUI2'])

# Remove missing or self-looped relations
df = df.dropna()
df = df[df['CUI1'] != df['CUI2']]

# Optional: filter certain relations (e.g., only "RB", "RN", "PAR")
allowed_rels = ['RB', 'RN', 'PAR']
df = df[df['REL'].isin(allowed_rels)]

# Create triplets: subject, predicate, object
triplets = list(zip(df['CUI1'], df['REL'], df['CUI2']))

# Save if needed
pd.DataFrame(triplets, columns=['subject', 'predicate', 'object']).to_csv('UMLS/umls_triples.csv', index=False)
