# Medical Records System

A patient medical records system that manages diagnoses and medications
backed by SQLite, using Advanced Python patterns for robustness and
NumPy for clinical analytics.

Built to consolidate Advanced Python, SQL, and NumPy in a single
healthcare-focused project.

---

## Features

- **Context Manager** — automatic commit/rollback/close on every database operation
- **Decorator Stacking** — input validation and operation logging applied together
- **Generator** — memory-efficient patient report yielded one line at a time
- **Custom Iterator** — medication schedule traversed with `__iter__` and `__next__`
- **LEFT JOIN** — patients without diagnoses or medications still appear in reports
- **COALESCE** — NULL values replaced with readable defaults at the SQL level
- **GROUP BY + AVG** — severity distribution with average patient age per tier
- **NumPy Analytics** — age statistics and boolean masking for risk classification

---

## Tech Stack

| Layer       | Technology                                         |
|-------------|----------------------------------------------------|
| Language    | Python 3.12                                        |
| Database    | SQLite (via built-in sqlite3)                      |
| Patterns    | Repository, Context Manager, Decorator, Generator  |
| Analytics   | NumPy 2.4.4                                        |

---

## Project Structure
```
medical-records/
├── medical_records.py    # All domain logic, SQL, and analytics
└── .gitignore
```
---

## How to Run

```bash
git clone https://github.com/volkancelebidev/medical-records.git
cd medical-records
pip install numpy
python medical_records.py
```

---

## What I Learned

- Implementing the context manager protocol with `__enter__` and `__exit__`
- Stacking decorators for validation and logging without code duplication
- Using generators with `yield` for memory-efficient data processing
- Building a custom iterator with `__iter__` and `__next__`
- Writing LEFT JOIN queries to preserve unmatched rows
- Applying COALESCE to handle NULL values cleanly in SQL
- Combining list comprehensions with NumPy for vectorised analytics
