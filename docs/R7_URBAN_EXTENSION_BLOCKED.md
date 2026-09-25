# R7 urban extension: why #9 stays BLOCKED (#9)

Issue #9 requires "real high-resolution dynamic targets (e.g. HRCLDAS/SMBFD/regional
NWP)" and states explicitly that "interpolated ERA5 is baseline only, never truth".
This page records what was actually checked, and why the dependency is real rather
than assumed.

**Verdict: BLOCKED.** No open, registration-free, km-scale **dynamic** analysis is
available for the R7 East-Asia ROI. This is not a statement that no such data
exists on Earth — it is a statement that none of the candidate routes reachable
from this machine provides it under #9's constraint.

`scientific_claim: false`. Nothing here is a result; it is an access audit.

## The constraint that decides it

#9 needs three things simultaneously:

1. **Co-located** with the R7 domain. The store actually used for the #5–#8 work
   spans **27.00–43.00 N, 107.00–123.00 E at 0.25°**, so any truth field must
   cover that box.
2. **Dynamic** (time-evolving analysed fields), not a static map or a land-surface
   climatology.
3. **Finer than 0.25°**, and genuinely so — upsampled 0.25° reanalysis is
   explicitly excluded by the issue.

## What was checked, and what each check found

| Candidate | Checked how | Result |
| --- | --- | --- |
| **HRRR** (NOAA, 3 km CONUS) | live GCS bucket listing, `noaa-hrrr-bdp-pds` | **Reachable and open (HTTP 200)**, but domains offered are **CONUS + Alaska only**. The R7 box is at 107–123 °E, outside CONUS. Not co-located. |
| **HRCLDAS** (CMA) | CMA portal `data.cma.cn`; OpenAlex literature search (39 works) | Products are described in the peer-reviewed literature, but the front page exposes **no HRCLDAS product listing and no machine-readable, key-free endpoint**. Access is via the CMA service portal, which is a registration/request route, not an anonymous object store. |
| **SMBFD** | OpenAlex literature search | Only **4 works** match, and they are *evaluations* of reanalysis products over the Tibetan Plateau and lake districts — not an open download of a km-scale dynamic analysis for East Asia. |
| **WeatherBench2** | live bucket index | 2 top-level prefixes; **no km / urban / China / East-Asia product**. |
| **ARCO ERA5** | live bucket index | 3 top-level prefixes; all are the known ERA5 reanalysis resolutions (**0.25°** and coarser). No sub-0.25° dynamic product. |
| **NOAA PSL gridded archive** | live dataset index | No `cma` / `china` / `hrcldas` / `smbfd` entry; the only km-scale regional product family present is the US one (HRRR/RAP/NAM). |
| Search engines | `cn.bing.com` via WebFetch | Returned **entirely unrelated results** (browser-game pages) for both HRCLDAS and SMBFD queries. Recorded here because it demonstrates that a search summary must not be treated as evidence — the access conclusion above rests on the direct endpoint checks, not on these. |

Every "reachable" claim above was tested against a live endpoint in this round;
every "not present" claim comes from an authoritative bucket or archive index
queried directly, not from a search snippet.

## Why not use the US 3 km product anyway?

Because it is not co-located with anything R7 has measured, and #9 is explicitly an
*urban extension of the R7 forecast* — routing selected regions of an East-Asia
0.25° forecast into a finer-resolution expert. A US-only truth field would require
either retraining the whole core on a CONUS domain (a different project) or
evaluating an East-Asia forecast against non-overlapping truth (meaningless).

## The failure mode this avoids

The tempting shortcut is to regrid 0.25° ERA5 to 1 km and call it truth. That is
exactly what #9 forbids, and it would be undetectable in a metric table: an
upsampled field has *smoother* gradients, so a model trained against it can score
well on small-scale error it never actually learned. Every R7 artifact already
records `spatial_resampling: none` / `interpolation: false` for this reason, and
no interpolation was performed here.

## What would unblock it

Any **one** of the following, all of which need a human decision or an outside
account rather than more compute:

1. **A CMA (or equivalent) data-service account** with HRCLDAS/CLDAS download
   rights, plus an explicit statement of the redistribution terms, so the subset
   can be fetched through `data/download/**` under the existing budget guards.
2. **A regionally co-located open km-scale analysis for 107–123 °E** — for example
   a national meteorological service publishing an anonymous object store.
   Japan's and Korea's operational archives were not reachable as key-free object
   stores in this round and would need the same review.
3. **Pivoting the study domain to CONUS**, which makes open 3 km HRRR truth
   immediately usable, at the cost of redoing the core forecast training on that
   domain — a scope decision, not an implementation task.

Until one of these exists, #9 has no honest path to a measurement, and the correct
state is BLOCKED with the dependency named. Fabricating the truth field is not an
option this project permits.

## What was NOT done

- No HRCLDAS or SMBFD file was downloaded, because no key-free endpoint was found.
- No registration or account was created (that is a user-side action).
- No interpolation, upsampling or synthetic stand-in was produced, and no metric
  was computed against anything that is not real reanalysis.
- The UrbanExpert component was not modified, and no claims about its performance
  are made.
