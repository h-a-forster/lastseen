# Meta models in LastSeen

Every model call goes through one gateway, [`app/meta_api.py`](../app/meta_api.py): retries on 429, 500 and 503
with backoff, a spending cap per process, a ledger line for every paid call, and a cache keyed by a hash of the
request, so tests, benchmarks and the demo replay answers for $0. Muse Spark always answers in strict JSON against a
schema, and code checks the answer before anything uses it ([Who decides what](../README.md#who-decides-what)).

## What each model does

| Model | Job | Called from | What we measured |
|---|---|---|---|
| **SAM 3.1** (`sam-3.1`) | Proposes every person in a 1000 px drone-photo tile from the text prompt "person", with a box and a confidence | `sam_find` in `app/meta_api.py`; [`app/perception/gate.py`](../app/perception/gate.py), [`triage.py`](../app/perception/triage.py) | On 784 tiles of real drone photos (660 labelled people), SAM alone finds 74% of them, with no training on our photos |
| **Muse Spark** (`muse-spark-1.3`) | Checks each SAM proposal on an enlarged close-up: person or not, and what it looks like | [`app/perception/verify.py`](../app/perception/verify.py) | Cuts the dataset's false alarms by 37% (0.103 to 0.065 per tile) and keeps 485 of the 492 people SAM found |
| | Reads the brief and every witness report into at most 5 weighted places to search, each quoting the words it rests on | [`app/agent/interpret.py`](../app/agent/interpret.py), through [`advisor.py`](../app/agent/advisor.py) | On 200 held-out searches: half found by 1 h 52 instead of 3 h 17, +6.5 points within 2 hours ([evaluation](evaluation.md)). With places named instead of distances, +14.6 points where rules gain nothing |
| | Grades each witness the way a search manager would (do they know her, did they see it themselves, does anyone back it up?) | [`app/agent/witnesses.py`](../app/agent/witnesses.py) | Shown next to the places, never used to change weights |
| | Words the handover to the ground team from facts code worked out | [`app/agent/handover.py`](../app/agent/handover.py) | Any number or direction that isn't one of the facts rejects the message |
| **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Turns a spoken brief or a witness's call into text | `transcribe` in `app/meta_api.py`; `/api/transcribe` in [`app/main.py`](../app/main.py) | Not benchmarked on its own: the transcript is typed into the text box, where the volunteer can correct it before Muse Spark reads it |
| **Muse Spark Open** (`meta-models/muse-spark-open-1026`, on Baseten) | Any of Spark's jobs on open weights (`SPARK_BACKEND=open`); on the page, "Use open weights" for a brief, and the question box for every visitor | `spark_model` in `app/meta_api.py`; [`app/faq.py`](../app/faq.py) | Half found by 2 h 05 instead of 3 h 17 on the same 200 searches; 177 of 265 people found on the photo check's test tiles, against 179 with Muse Spark ([results](results/final-open-weights.txt)) |

Each job's instructions are the `SYSTEM` string and its answer format the `SCHEMA` at the top of the file named.
Spark runs at `reasoning_effort` "low" everywhere: each job is one bounded reading, and reasoning is billed as output.

## Calls and cost
From the spend ledger, every paid call of the hackathon, at dev.meta.ai and Baseten list prices:

| Model | Paid calls | Cost | Per call |
|---|---|---|---|
| Muse Spark | 2,288 | $14.00 | $0.006 |
| SAM 3.1 | 821 | $2.05 | $0.0025 (billed per image) |
| Muse Spark Open | 871 | $4.16 | $0.005 |
| Muse Voice Transcribe | 8 | under $0.01 | $0.0005 (billed per hour of audio) |

That covers building, the detection measurements and every evaluation run; the hosted page's own calls add about
$0.35 so far. A rerun of any of them costs nothing, because every answer is cached.

**Limits on the hosted page:** one paid job at a time (others get "busy, try again in a few seconds"), at most 60
paid requests an hour per visitor address, and a $20 cap on the server's total model spend. The recorded races,
deep links and examples never call a model, so they keep working whatever the limits say.

## Models we didn't use
**Muse Wonder.** Meta positions Wonder for "long-horizon agentic work", such as changes across many files and
multi-hour agent runs, and Muse Spark for high-volume, latency-sensitive work such as classification and tight tool
loops ([Meta's model guide](https://dev.meta.ai/docs/models)). Every model job in LastSeen is of the second kind:
one bounded reading, returned as JSON that code checks, up to 30 times for a single photo. Wonder also costs about three times
as much per input token and nearly five times per output token ($4 and $20 per million, against $1.25 and $4.25).

The one long-horizon job in a search is choosing the route, and that is code on purpose: the same map must give the
same route, a new leg must be ready mid-flight without the drone waiting, and the planner had to be compared on 200
searches fixed in advance, which a model choosing every leg would make slow, costly and impossible to repeat exactly.
Where Wonder would fit is offline: writing and testing new versions of the planner's code against the simulator,
which runs a 4-hour search in under a second with no model calls, and keeping only versions that beat the current
planner on held-out searches.

**Muse Image.** Every find is decided on a real drone photo of a person that size, never on a generated one, so the
search has nothing for an image generator to do.
