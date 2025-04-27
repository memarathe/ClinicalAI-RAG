CREATE TABLE cost (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    event_type TEXT, -- Type of event
    event_id INTEGER, -- Unique identifier for each event
    chargetime TIME, -- Time of event
    cost INTEGER, -- Cost of event
);

CREATE TABLE diagnoses_icd (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    icd_code TEXT, -- ICD code for diagnosis
    charttime TIME, -- Time of diagnosis
);

CREATE TABLE procedures_icd (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    icd_code TEXT, -- ICD code for procedure
    charttime TIME, -- Time of procedure
);

CREATE TABLE labevents (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    itemid INTEGER, -- Unique identifier for each lab test
    charttime TIME, -- Time of lab test
    valuenum INTEGER, -- value of lab test
    valueuom TEXT, -- unit of measurement for lab test
);

CREATE TABLE prescriptions (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    starttime TIME, -- Prescribed start time for drug
    stoptime TIME, -- Prescribed stop time for drug
    drug TEXT, -- Description of medication
    dose_val_rx TEXT, -- Prescribed dose of drug
    dose_unit_rx TEXT, -- Unit of measurement for drug dose
    route TEXT, -- Route of administration for drug
);

CREATE TABLE chartevents (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    stay_id INTEGER, -- Unique identifier for each care unit stay
    itemid INTEGER, -- Unique identifier for each measurement
    charttime TIME, -- Time at which measurement was taken
    valuenum INTEGER, -- Value measured for concept identified by itemid
    valueuom TEXT, -- Unit of measurement for the value
);

CREATE TABLE inputevents (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    stay_id INTEGER, -- Unique identifier for each care unit stay
    starttime TIME, -- Start time of input event
    itemid INTEGER, -- Unique identifier for each measurement
    amount INTEGER, -- Amount of drug
);

CREATE TABLE outputevents (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    stay_id INTEGER, -- Unique identifier for each care unit stay
    charttime TIME, -- Time of output event
    itemid INTEGER, -- Unique identifier for each measurement
    value INTEGER, -- Amount of output
);

CREATE TABLE microbiologyevents (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    charttime TIME, -- Time at which measurement was taken
    spec_type_desc TEXT, -- Description of specimen type
    test_name TEXT, -- Name of test performed
    org_name TEXT, -- Name of organism which grew when test was performed
);

CREATE TABLE icustays (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    stay_id INTEGER, -- Unique identifier for each care unit stay
    first_careunit TEXT, -- First care unit for patient
    last_careunit TEXT, -- Last care unit for patient
    intime TIME, -- Time patient transferred to ICU
    outtime TIME, -- Time patient transferred out of ICU
);

CREATE TABLE transfers (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    transfer_id INTEGER, -- Unique identifier for patient location
    eventtype TEXT, -- Type of transfer
    careunit TEXT, -- Type of unit
    intime TIME, -- Time patient transferred into care unit
    outtime TIME, -- Time patient transferred out of care unit
);

CREATE TABLE admissions (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    hadm_id INTEGER, -- Unique identifier for each hospital admission
    admittime TIME, -- Time of admission
    dischtime TIME, -- Time of discharge
    admission_type TEXT, -- Type of admission
    admission_location TEXT, -- Location of patient before admission
    discharge_location TEXT, -- Location of patient after discharge
    insurance TEXT, -- Name of insurance
    language TEXT, -- Language of patient
    marital_status TEXT, -- Marital status of patient
    age INTEGER, -- Age of patient
);

CREATE TABLE patients (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    subject_id INTEGER, -- Unique identifier for each patient
    gender TEXT, -- Gender of patient
    dob TIME, -- Date of birth of patient
    dod TIME, -- Date of death of patient
);

CREATE TABLE d_icd_diagnoses (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    icd_code TEXT, -- ICD code for diagnosis
    long_title TEXT, -- Description of diagnosis
);

CREATE TABLE d_icd_procedures (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    icd_code TEXT, -- ICD code for procedure
    long_title TEXT, -- Description of procedure
);

CREATE TABLE d_labitems (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    itemid INTEGER, -- Unique identifier for each lab test
    label TEXT, -- Name of lab test
);

CREATE TABLE d_items (
    row_id INTEGER PRIMARY KEY, -- Unique identifier for each row
    itemid INTEGER, -- Unique identifier for each item
    label TEXT, -- Concept represented by item
    abbreviation TEXT, -- Abbreviation of label
    linksto TEXT, -- table name that item links to
);

-- diagnoses_icd.event_id can be joined with cost.row_id
-- procedures_icd.event_id can be joined with cost.row_id
-- labevents.event_id can be joined with cost.row_id
-- prescriptions.event_id can be joined with cost.row_id
-- icustays.stay_id can be joined with chartevents.stay_id
-- icustays.stay_id can be joined with inputevents.stay_id
-- icustays.stay_id can be joined with outputevents.stay_id
-- admissions.hadm_id can be joined with diagnoses_icd.hadm_id
-- admissions.hadm_id can be joined with procedures_icd.hadm_id
-- admissions.hadm_id can be joined with labevents.hadm_id
-- admissions.hadm_id can be joined with prescriptions.hadm_id
-- admissions.hadm_id can be joined with cost.hadm_id
-- admissions.hadm_id can be joined with chartevents.hadm_id
-- admissions.hadm_id can be joined with inputevents.hadm_id
-- admissions.hadm_id can be joined with outputevents.hadm_id
-- admissions.hadm_id can be joined with microbiologyevents.hadm_id
-- admissions.hadm_id can be joined with icustays.hadm_id
-- admissions.hadm_id can be joined with transfers.hadm_id
-- patients.subject_id can be joined with admissions.subject_id
-- d_icd_diagnoses.icd_code can be joined with diagnoses_icd.icd_code
-- d_icd_procedures.icd_code can be joined with procedures_icd.icd_code
-- d_labitems.itemid can be joined with labevents.itemid
-- d_items.itemid can be joined with chartevents.itemid
-- d_items.itemid can be joined with inputevents.itemid
-- d_items.itemid can be joined with outputevents.itemid