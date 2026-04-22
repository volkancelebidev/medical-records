"""
medical_records.py

A patient medical records system that manages diagnoses and medications
backed by SQLite, using Advanced Python patterns for robustness and
NumPy for clinical analytics.

Covered topics:
    Advanced Python → decorator (stacking), generator, iterator, context manager
    SQL             → multi-table schema, LEFT JOIN, GROUP BY, COALESCE, transaction
    NumPy           → array creation, statistical functions, boolean masking

Typical usage:
    python medical_records.py
"""

import sqlite3
import numpy as np
from datetime import datetime
from functools import wraps


# ---------------------------------------------------------------------------
# Advanced Python — Decorators
# ---------------------------------------------------------------------------

def validate_input(func):
    """Reject calls where the first string argument is empty or whitespace.

    Applied to write operations so invalid data never reaches the database.
    @wraps preserves the wrapped function's __name__ and __doc__ so
    introspection and logging still show the original function name.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function that raises ValueError on empty string input.
    """
    @wraps(func)    # wraps → keep original function name and docstring
    def wrapper(*args, **kwargs):
        # args[1] is the first caller argument (args[0] is self).
        if args and isinstance(args[1], str) and not args[1].strip():
            raise ValueError(f"Empty string passed to {func.__name__}()")
        return func(*args, **kwargs)
    return wrapper


def log_operation(func):
    """Log every database write with a timestamp.

    Stacked with @validate_input so each write operation is both
    validated and logged without duplicating that logic.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function that prints a log line after each successful call.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)   # run the original function first
        print(f"[LOG] {datetime.now().strftime('%H:%M:%S')} — {func.__name__}() executed")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Advanced Python — Context Manager
# ---------------------------------------------------------------------------

class DatabaseConnection:
    """Managed SQLite connection that commits on success and rolls back on error.

    Implements the context manager protocol so callers never need to
    handle commit/rollback/close manually.

    Usage:
        with DatabaseConnection("db.sqlite") as conn:
            conn.execute("INSERT ...")

    __enter__ opens and returns the connection.
    __exit__  commits on clean exit, rolls back on exception, always closes.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn    = None   # no connection yet — opened in __enter__

    def __enter__(self) -> sqlite3.Connection:
        # __enter__ → called when entering the with block
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        return self.conn   # bound to the `as` variable

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # __exit__ → called when leaving the with block, even on exception
        if exc_type:
            self.conn.rollback()   # something went wrong — undo all changes
        else:
            self.conn.commit()     # clean exit — persist changes
        self.conn.close()          # always release the connection


# ---------------------------------------------------------------------------
# Advanced Python — Generator
# ---------------------------------------------------------------------------

def patient_report_generator(patients: list[dict]):
    """Yield one formatted report line per patient.

    Using a generator instead of building the full list avoids holding
    all formatted strings in memory simultaneously — important when
    processing thousands of records.

    Args:
        patients: List of patient dicts, each containing at least
                  patient_id, name, age, and diagnosis keys.

    Yields:
        One formatted string per patient.
    """
    for patient in patients:   # patient → each dict in the list
        # yield → produce one value and pause; resume on next iteration
        yield (
            f"ID: {patient['patient_id']:>5} | "
            f"{patient['name']:<20} | "
            f"Age: {patient['age']:>3} | "
            f"Diagnosis: {patient['diagnosis']}"
        )


# ---------------------------------------------------------------------------
# Advanced Python — Iterator
# ---------------------------------------------------------------------------

class MedicationSchedule:
    """Iterate over a patient's medication list one drug at a time.

    Implements the iterator protocol manually to demonstrate how Python's
    for loop works under the hood.

    __iter__ returns the iterator object itself.
    __next__ returns the next item or raises StopIteration when exhausted.
    """

    def __init__(self, medications: list[str]) -> None:
        self.medications = medications
        self._index      = 0   # _ prefix → internal state, not part of public API

    def __iter__(self):
        # __iter__ → called by for loops to get the iterator object
        return self

    def __next__(self) -> str:
        # __next__ → called on each loop iteration
        if self._index >= len(self.medications):
            raise StopIteration   # signal the for loop to stop
        med = self.medications[self._index]
        self._index += 1   # advance the cursor for the next call
        return med


# ---------------------------------------------------------------------------
# Database — Repository Pattern
# ---------------------------------------------------------------------------

class MedicalRecordsDB:
    """Centralised data access layer for patients, diagnoses, and medications.

    Follows the Repository pattern: all SQL lives here so domain logic
    elsewhere never touches the database directly.

    Args:
        db_path: Path to the SQLite file. Created on first run.
    """

    def __init__(self, db_path: str = "medical_records.db") -> None:
        self.db_path = db_path
        self._setup_schema()   # create tables on first run

    def _setup_schema(self) -> None:
        """Create (or recreate) the three-table schema.

        executescript() runs each statement in autocommit mode, which is
        required for PRAGMA foreign_keys to take effect reliably.

        Drop order matters: dependent tables (diagnoses, medications) must
        be removed before the tables they reference (patients).
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.executescript("""
                PRAGMA foreign_keys = ON;

                DROP TABLE IF EXISTS medications;
                DROP TABLE IF EXISTS diagnoses;
                DROP TABLE IF EXISTS patients;

                CREATE TABLE patients (
                    patient_id   TEXT PRIMARY KEY,
                    name         TEXT NOT NULL,
                    age          INTEGER NOT NULL,
                    blood_type   TEXT NOT NULL,
                    admitted_at  TEXT NOT NULL
                );

                CREATE TABLE diagnoses (
                    diagnosis_id  TEXT PRIMARY KEY,
                    patient_id    TEXT NOT NULL REFERENCES patients(patient_id),
                    diagnosis     TEXT NOT NULL,
                    severity      TEXT NOT NULL,
                    diagnosed_at  TEXT NOT NULL
                );

                CREATE TABLE medications (
                    med_id      TEXT PRIMARY KEY,
                    patient_id  TEXT NOT NULL REFERENCES patients(patient_id),
                    drug_name   TEXT NOT NULL,
                    dosage      TEXT NOT NULL,
                    frequency   TEXT NOT NULL,
                    start_date  TEXT NOT NULL
                );
            """)

    # ------------------------------------------------------------------
    # Patient operations
    # ------------------------------------------------------------------

    @log_operation      # log_operation → log after execution
    @validate_input     # validate_input → reject empty strings before running
    def add_patient(
        self,
        patient_id: str,
        name: str,
        age: int,
        blood_type: str,
    ) -> None:
        """Register a new patient.

        Decorators are applied bottom-up: validate_input runs first,
        then the function executes, then log_operation logs the result.
        INSERT OR IGNORE silently skips duplicate patient_id values.

        Args:
            patient_id: Unique identifier (PRIMARY KEY).
            name:       Full name of the patient.
            age:        Age in years.
            blood_type: ABO/Rh blood group (e.g. "A+").
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO patients
                    (patient_id, name, age, blood_type, admitted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (patient_id, name, age, blood_type,
                 datetime.now().strftime("%Y-%m-%d")),
            )

    def get_all_patients(self) -> list[dict]:
        """Return all patients ordered alphabetically.

        row_factory converts sqlite3.Row objects to plain dicts so callers
        can use familiar key-based access instead of positional indexing.

        Returns:
            List of patient dicts ordered by name.
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.row_factory = sqlite3.Row   # row_factory → dict-like row access
            rows = conn.execute(
                "SELECT * FROM patients ORDER BY name"
            ).fetchall()
        return [dict(row) for row in rows]   # dict(row) → convert Row to dict

    # ------------------------------------------------------------------
    # Diagnosis operations
    # ------------------------------------------------------------------

    @log_operation
    def add_diagnosis(
        self,
        diagnosis_id: str,
        patient_id: str,
        diagnosis: str,
        severity: str,
    ) -> None:
        """Record a diagnosis for an existing patient.

        The REFERENCES constraint on patient_id prevents orphaned diagnoses —
        inserting a non-existent patient_id raises IntegrityError.

        Args:
            diagnosis_id: Unique identifier for this diagnosis record.
            patient_id:   Must reference an existing patients.patient_id.
            diagnosis:    Clinical diagnosis description.
            severity:     One of "Mild", "Moderate", "Critical".
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO diagnoses
                    (diagnosis_id, patient_id, diagnosis, severity, diagnosed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (diagnosis_id, patient_id, diagnosis, severity,
                 datetime.now().strftime("%Y-%m-%d")),
            )

    def get_patient_diagnoses(self, patient_id: str) -> list[dict]:
        """Return all diagnoses for a patient, newest first.

        Parameterised query (?) prevents SQL injection — the patient_id
        value is never interpolated directly into the SQL string.

        Args:
            patient_id: The patient whose diagnoses to retrieve.

        Returns:
            List of diagnosis dicts ordered by date descending.
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM diagnoses
                WHERE patient_id = ?
                ORDER BY diagnosed_at DESC
                """,
                (patient_id,),   # trailing comma → single-element tuple
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Medication operations
    # ------------------------------------------------------------------

    @log_operation
    def add_medication(
        self,
        med_id: str,
        patient_id: str,
        drug_name: str,
        dosage: str,
        frequency: str,
    ) -> None:
        """Prescribe a medication to an existing patient.

        Args:
            med_id:     Unique identifier for this prescription record.
            patient_id: Must reference an existing patients.patient_id.
            drug_name:  Name of the prescribed drug.
            dosage:     Dose per administration (e.g. "10mg").
            frequency:  Dosing schedule (e.g. "Twice daily").
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO medications
                    (med_id, patient_id, drug_name, dosage, frequency, start_date)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (med_id, patient_id, drug_name, dosage, frequency,
                 datetime.now().strftime("%Y-%m-%d")),
            )

    def get_patient_medications(self, patient_id: str) -> list[dict]:
        """Return all active medications for a patient.

        Args:
            patient_id: The patient whose medications to retrieve.

        Returns:
            List of medication dicts.
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM medications WHERE patient_id = ?",
                (patient_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # JOIN queries
    # ------------------------------------------------------------------

    def get_full_records(self) -> list[dict]:
        """Return patients joined with their diagnoses and medications.

        LEFT JOIN ensures patients without a diagnosis or medication still
        appear in the results.  COALESCE replaces NULL with a readable
        default so callers receive clean strings, not None values.

        Returns:
            List of combined record dicts ordered by patient name.
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    p.patient_id,
                    p.name,
                    p.age,
                    p.blood_type,
                    COALESCE(d.diagnosis, 'No diagnosis') AS diagnosis,
                    COALESCE(d.severity,  'N/A')          AS severity,
                    COALESCE(m.drug_name, 'No medication') AS medication,
                    COALESCE(m.dosage,    'N/A')           AS dosage
                FROM patients p
                LEFT JOIN diagnoses   d ON p.patient_id = d.patient_id
                LEFT JOIN medications m ON p.patient_id = m.patient_id
                ORDER BY p.name
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_severity_stats(self) -> list[dict]:
        """Aggregate diagnosis counts and average patient age by severity.

        GROUP BY collapses all rows with the same severity into one summary
        row.  AVG(p.age) computes the mean age across that group.

        Returns:
            List of dicts with severity, count, and avg_age,
            ordered by count descending.
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    d.severity,
                    COUNT(*)   AS count,
                    AVG(p.age) AS avg_age
                FROM diagnoses d
                INNER JOIN patients p ON d.patient_id = p.patient_id
                GROUP BY d.severity
                ORDER BY count DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # NumPy analytics
    # ------------------------------------------------------------------

    def age_analysis(self) -> dict:
        """Compute descriptive statistics for patient ages using NumPy.

        List comprehension extracts ages from the patient dicts, then
        np.array() converts the list so vectorised functions can run
        without a Python loop.

        Returns:
            Dict with mean, median, std, min, max.
        """
        patients = self.get_all_patients()
        if not patients:
            return {}

        # list comprehension → extract age from each patient dict
        ages = np.array([p["age"] for p in patients])   # p → each patient dict

        return {
            "mean"  : round(float(np.mean(ages)),   1),
            "median": round(float(np.median(ages)), 1),
            "std"   : round(float(np.std(ages)),    1),
            "min"   : int(np.min(ages)),
            "max"   : int(np.max(ages)),
        }

    def risk_analysis(self) -> dict:
        """Count diagnoses in each severity tier using boolean masking.

        severities == "Critical" produces a boolean array; np.sum() counts
        the True values — no loop required.

        Returns:
            Dict with total_diagnoses, critical, moderate, and mild counts.
        """
        with DatabaseConnection(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT severity FROM diagnoses").fetchall()

        if not rows:
            return {}

        # list comprehension → extract severity string from each row
        severities = np.array([r["severity"] for r in rows])   # r → each row

        return {
            "total_diagnoses": len(severities),
            "critical"       : int(np.sum(severities == "Critical")),  # boolean mask
            "moderate"       : int(np.sum(severities == "Moderate")),  # boolean mask
            "mild"           : int(np.sum(severities == "Mild")),      # boolean mask
        }

    def medication_load(self) -> dict:
        """Compute per-patient medication counts with NumPy.

        Returns:
            Dict with avg and max medications per patient.
        """
        patients = self.get_all_patients()
        if not patients:
            return {}

        # list comprehension → count medications for each patient
        counts = np.array([
            len(self.get_patient_medications(p["patient_id"]))
            for p in patients   # p → each patient dict
        ])

        return {
            "avg_per_patient": round(float(np.mean(counts)), 1),
            "max_per_patient": int(np.max(counts)),
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_full_report(db: MedicalRecordsDB) -> None:
    """Print all patient records as a formatted table.

    Args:
        db: Initialised MedicalRecordsDB instance.
    """
    print("\n[Full Medical Records — LEFT JOIN across 3 tables]")
    separator = "-" * 84
    print(separator)
    print(f"  {'ID':<8} {'Name':<18} {'Age':>4} {'Blood':>6} "
          f"{'Diagnosis':<22} {'Severity':<10} {'Medication':<16} Dosage")
    print(separator)

    for r in db.get_full_records():   # r → each record dict
        print(
            f"  {r['patient_id']:<8} "
            f"{r['name']:<18} "
            f"{r['age']:>4} "
            f"{r['blood_type']:>6} "
            f"{r['diagnosis']:<22} "
            f"{r['severity']:<10} "
            f"{r['medication']:<16} "
            f"{r['dosage']}"
        )
    print(separator)


def print_analytics(db: MedicalRecordsDB) -> None:
    """Print NumPy-powered analytics and SQL aggregate results.

    Args:
        db: Initialised MedicalRecordsDB instance.
    """
    sep = "-" * 44

    print("\n[Age Statistics — NumPy]")
    print(sep)
    for key, val in db.age_analysis().items():   # key, val → dict key-value pairs
        print(f"  {key:<8}: {val}")
    print(sep)

    print("\n[Diagnosis Risk Analysis — Boolean Masking]")
    print(sep)
    for key, val in db.risk_analysis().items():
        print(f"  {key:<20}: {val}")
    print(sep)

    print("\n[Severity Distribution — GROUP BY + AVG]")
    print(sep)
    for row in db.get_severity_stats():   # row → each stats dict
        print(
            f"  {row['severity']:<12} | "
            f"Count: {row['count']:>2} | "
            f"Avg Age: {row['avg_age']:.1f}"
        )
    print(sep)

    print("\n[Medication Load — NumPy]")
    print(sep)
    for key, val in db.medication_load().items():
        print(f"  {key:<22}: {val}")
    print(sep)


def print_generator_report(db: MedicalRecordsDB) -> None:
    """Demonstrate the generator by printing one report line per patient.

    The generator yields lines on demand rather than building the full
    list in memory — scalable to large patient populations.

    Args:
        db: Initialised MedicalRecordsDB instance.
    """
    patients = db.get_all_patients()

    # Build combined records by attaching the latest diagnosis to each patient.
    patient_records = []
    for p in patients:   # p → each patient dict
        diagnoses      = db.get_patient_diagnoses(p["patient_id"])
        # ternary operator → use first diagnosis if available, else default string
        diagnosis_text = diagnoses[0]["diagnosis"] if diagnoses else "No diagnosis"
        # {**p, "diagnosis": ...} → copy p and add the diagnosis key
        patient_records.append({**p, "diagnosis": diagnosis_text})

    print("\n[Generator Report — yielded one line at a time]")
    print("-" * 64)

    # patient_report_generator → create the generator object
    report = patient_report_generator(patient_records)

    for line in report:   # line → each string yielded by the generator
        print(f"  {line}")

    print("-" * 64)


def print_iterator_example(db: MedicalRecordsDB) -> None:
    """Demonstrate the custom iterator by walking one patient's medications.

    Args:
        db: Initialised MedicalRecordsDB instance.
    """
    patients = db.get_all_patients()
    if not patients:
        return

    first = patients[0]   # patients[0] → first patient in the list
    meds  = db.get_patient_medications(first["patient_id"])
    drug_names = [m["drug_name"] for m in meds]   # m → each medication dict

    if not drug_names:
        return

    print(f"\n[Medication Iterator — {first['name']}]")
    print("-" * 40)

    # MedicationSchedule → instantiate the custom iterator
    schedule = MedicationSchedule(drug_names)

    for med in schedule:   # med → value returned by __next__ on each iteration
        print(f"  → {med}")

    print("-" * 40)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Seed the database with sample data and run all reports."""

    print("[+] Medical Records System — starting\n")

    db = MedicalRecordsDB("medical_records.db")

    # --- Patients ----------------------------------------------------------
    patient_data = [
        ("P001", "Alice Kaya",    45, "A+"),
        ("P002", "Mehmet Demir",  62, "B-"),
        ("P003", "Selin Arslan",  34, "O+"),
        ("P004", "Burak Gunes",   71, "AB+"),
        ("P005", "Zeynep Toprak", 28, "A-"),
    ]

    # tuple unpacking → pid, name, age, blood from each tuple
    for pid, name, age, blood in patient_data:
        db.add_patient(pid, name, age, blood)

    # --- Diagnoses ---------------------------------------------------------
    diagnosis_data = [
        ("D001", "P001", "Hypertension",    "Moderate"),
        ("D002", "P002", "Type 2 Diabetes", "Critical"),
        ("D003", "P003", "Migraine",         "Mild"),
        ("D004", "P004", "Heart Failure",    "Critical"),
        ("D005", "P005", "Anxiety Disorder", "Mild"),
    ]

    for did, pid, diag, sev in diagnosis_data:
        db.add_diagnosis(did, pid, diag, sev)

    # --- Medications -------------------------------------------------------
    medication_data = [
        ("M001", "P001", "Lisinopril",  "10mg",    "Once daily"),
        ("M002", "P001", "Amlodipine",  "5mg",     "Once daily"),
        ("M003", "P002", "Metformin",   "500mg",   "Twice daily"),
        ("M004", "P002", "Insulin",     "10U",     "Before meals"),
        ("M005", "P003", "Sumatriptan", "50mg",    "As needed"),
        ("M006", "P004", "Furosemide",  "40mg",    "Once daily"),
        ("M007", "P004", "Carvedilol",  "6.25mg",  "Twice daily"),
        ("M008", "P005", "Sertraline",  "50mg",    "Once daily"),
    ]

    for mid, pid, drug, dose, freq in medication_data:
        db.add_medication(mid, pid, drug, dose, freq)

    # --- Reports -----------------------------------------------------------
    print_full_report(db)
    print_analytics(db)
    print_generator_report(db)
    print_iterator_example(db)


if __name__ == "__main__":
    # Runs only when executed directly — not when imported as a module.
    main()