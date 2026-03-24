-- agentkit demo: real estate property intelligence

CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT NOT NULL,
    city TEXT,
    state TEXT,
    zipcode TEXT,
    lat REAL,
    lon REAL,
    bedrooms INTEGER,
    bathrooms REAL,
    sqft INTEGER,
    lot_sqft INTEGER,
    year_built INTEGER,
    property_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id INTEGER REFERENCES properties(id),
    address TEXT NOT NULL,
    filename TEXT NOT NULL,
    doc_type TEXT,  -- deed, inspection, hoa, appraisal
    content_text TEXT,
    extracted_json TEXT DEFAULT '{}',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_query TEXT NOT NULL,
    answer TEXT,
    grounding_score REAL,
    agent_trace_json TEXT DEFAULT '[]',
    total_cost REAL DEFAULT 0.0,
    latency_s REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS comparables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    address TEXT,
    city TEXT,
    state TEXT,
    zipcode TEXT,
    county TEXT,
    subdivision TEXT,
    lot_number TEXT,
    acreage REAL,
    property_type TEXT DEFAULT 'Vacant Land',
    sale_date TEXT,
    sale_price REAL,
    price_per_acre REAL,
    market_value REAL,
    appraised_value REAL,
    has_improvements INTEGER DEFAULT 0,
    improvement_value REAL DEFAULT 0,
    data_source TEXT,  -- 'blanco_cad', 'rentcast', 'manual'
    notes TEXT DEFAULT '',
    metadata_json TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_comparables_subdivision ON comparables(subdivision);
CREATE INDEX IF NOT EXISTS idx_comparables_zipcode ON comparables(zipcode);
CREATE INDEX IF NOT EXISTS idx_comparables_county ON comparables(county);

CREATE INDEX IF NOT EXISTS idx_properties_address ON properties(address);
CREATE INDEX IF NOT EXISTS idx_properties_zipcode ON properties(zipcode);
CREATE INDEX IF NOT EXISTS idx_documents_address ON documents(address);
CREATE INDEX IF NOT EXISTS idx_queries_session ON queries(session_id);
