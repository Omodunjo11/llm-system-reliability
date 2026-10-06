# Adversarial Audit Log

Results from deliberately breaking the reliability pipeline, then hardening it.

## Method

1. Run the golden RAG harness (`python -m src.eval_harness`)
2. Probe edge cases interactively
3. Run an adversarial battery designed to make the system **answer when it should abstain**
4. Record failures, fix gates, re-run until both suites pass

## Breaks found (pre-hardening)

| Case | Query pattern | Failure mode | Why it cleared the gate |
|------|---------------|--------------|-------------------------|
| `adv_basel_banks` | Basel capital + banks | Answered BSA text | Common term `banks` / `requirements` produced BM25 hits; rare term `Basel` unsupported |
| `adv_compare_basel_bsa` | Half-supported compare | Answered BSA/SAR only | Matched `bsa`/`sar` while `Basel`/`leverage` missing |
| `adv_ignore_invent` | Invent Reg Omega | Answered Reg FD | Lexical hits on `reg`/`regulation`/`aml` |
| `adv_fake_section` | BSA §99.9 quantum wallets | Answered BSA overview | Matched `bank secrecy act`; OOV `quantum`/`wallets` ignored |

# Lesson: Confidence that is a soft average of signals will still clear 0.5 when *common* overlap is strong. Distinctive unsupported query mass must be a **hard abstain**, not a mild penalty.

## Over-refusal after first harden (second break cycle)

| Case | Symptom | Fix |
|------|---------|-----|
| KYC definition | Abstain on missing rare term `stand` | Treat `stand` / format words as discourse stopwords |
| Insider trading "in one sentence" | Abstain on `one`,`sentence` | Strip instruction/format tokens from query terms |
| Single token `banks` | Answered with full coverage | Hard abstain when content terms < 2 (underspecified) |

## Hardening applied

1. **Discourse-stopword filtering** in query terms (`tell`, `compare`, `cite`, `ignore`, `invent`, …) so imperatives do not create false BM25 mass
2. **IDF-weighted coverage** using the BM25 index IDF — rare/OOV query terms weigh more than `banks`
3. **Hard abstain gate** when:
   - IDF-weighted coverage < 0.55, or
   - raw term coverage < 0.45, or
   - any unmatched query term is rare relative to the query's max IDF
4. Pipeline now passes `index.idf` into confidence estimation

## Post-hardening verification

```bash
python -m src.eval_harness --all   # golden + adversarial
python run.py audit
pytest -q
```

Latest run (local):

| Suite | Result |
|-------|--------|
| Golden RAG harness | 8/8 PASS |
| Adversarial harness | 10/10 PASS |
| pytest | 14 passed |

## What this still is / is not

- **Is:** a stdlib-only, auditable trust layer with measurable abstention under adversarial lexical overlap, plus a regression suite that encodes the breaks we found
- **Is not:** calibrated probabilistic uncertainty from a trained model, nor semantic entailment (NLI). Next upgrades would add hybrid retrieval or claim-level NLI without abandoning the hard-gate philosophy
