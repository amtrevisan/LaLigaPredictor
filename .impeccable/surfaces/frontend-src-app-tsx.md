---
version: 1
slug: "frontend-src-app-tsx"
primary_target: "frontend/src/App.tsx"
related_targets: []
---

# Predictor app surface

Scope: the whole web app (predictions, results, model pages) across six leagues. Visitor mode: Operate, with the Model page read as Read.
Audience and job: recruiters judging ML and full-stack ability in minutes; fans checking the next round. Proof: real pipeline output only.
Constraint from the user: Datamatics world, with more imagery — club crests and photos of past meetings between the two teams.

## Direction contract

THESIS: Every fixture is a data column. The round reads as one black field of hairline probability bars and dense tabular numbers, refusing the category default of rounded dark cards with a green accent.

OWN-WORLD: Pure black and pure white, one monospace face at data density, 1px hairline rules, bars for probabilities. Photography is admitted only through a 1-bit ordered-dither filter. The only colour on the page belongs to the clubs: crests in their own colours, and the focused fixture's photos.

STORY: The visitor sees this round's predictions immediately, understands home/draw/away at a glance, opens a fixture to see crests, the previous meetings in pictures, expected goals and the likeliest scorelines, then checks the Model page for out-of-sample performance against the market.

FIRST VIEWPORT: Top bar with wordmark, Predictions/Results/Model tabs, language switch and last-updated timestamp. League strip beneath. Main area: the round as a field of fixture columns (crest pair, short codes, stacked 1X2 bar, numbers); right rail: season log loss, accuracy and RPS against market and base rate. Selecting a column inverts it and opens the detail band below with dithered past-meeting photos that resolve to colour.

FORM: challenger ikeda-datamatics, adopted from the dealt hand over the assigned Sticker Album and the pick Pools Coupon; signature move is the column inversion with dither-to-colour reveal. seed:02b63f702756827f

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
