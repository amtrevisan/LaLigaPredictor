---
name: Match/Field
description: Six leagues of football predictions as one black field of hairline probability columns.
colors:
  ink: "#000000"
  paper: "#ffffff"
  dim: "#a3a3a3"
  rule: "#ffffff"
  rule-faint: "#2b2b2b"
typography:
  display:
    fontFamily: "Martian Mono Variable, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "2.5rem"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.03em"
    fontVariation: "'wdth' 80"
  headline:
    fontFamily: "Martian Mono Variable, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "1.75rem"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.02em"
    fontVariation: "'wdth' 80"
  title:
    fontFamily: "Martian Mono Variable, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "1.125rem"
    fontWeight: 700
    lineHeight: 1.5
  data:
    fontFamily: "Martian Mono Variable, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0.02em"
    fontFeature: "'tnum'"
    fontVariation: "'wdth' 80"
  body:
    fontFamily: "Martian Mono Variable, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "'tnum'"
  label:
    fontFamily: "Martian Mono Variable, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.75rem"
    fontWeight: 600
    lineHeight: 1.5
    letterSpacing: "0.06em"
rounded:
  none: "0px"
spacing:
  "1": "4px"
  "2": "8px"
  "3": "12px"
  "4": "16px"
  "5": "24px"
  "6": "32px"
  "7": "48px"
  "8": "64px"
  gutter: "clamp(16px, 2.5vw, 32px)"
  rail: "300px"
components:
  button-primary:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "8px 16px"
  tab:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.dim}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "0 16px"
  tab-hover:
    textColor: "{colors.paper}"
  tab-active:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
  select:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "4px 32px 4px 8px"
  fixture-column:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "12px 8px"
  fixture-column-selected:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
  tag-hit:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    padding: "1px 8px"
  tag-miss:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    typography: "{typography.label}"
    padding: "1px 8px"
---

# Design System: Match/Field

## Overview

**Creative North Star: "The Matchfield"**

One field of black, one figure of white, one monospace face. Every fixture is a data column: the round reads as a single black field of hairline rules, stacked outcome bars and dense tabular numbers, the way a stadium scoreboard or a printed coupon reads, not as a stack of rounded dark cards. The page is drawn, not painted: structure comes from 1px rules, values come from bar lengths and pattern, and hierarchy comes from weight, width and inversion.

Colour is withheld from the interface entirely so that it can belong to the clubs. Crests appear in their own colours at all times; club photography is held as a 1-bit ordered dither and resolves to colour only when its fixture is selected or the image is pointed at. That reveal is the world's one luxury, and it lands harder because nothing else on the page is coloured.

Density is high and deliberate. Body text sits at 13px in a monospace with tabular figures, and data compresses onto the font's width axis rather than shrinking. Motion is short and mechanical: wipes, never fades or bounces, except where a dithered layer clears.

**Key Characteristics:**
- Pure black field (#000) with pure white figure (#fff); one grey step for secondary text.
- One typeface, Martian Mono Variable, using weight and the width axis (80% for data, 112.5% for the wordmark).
- Home / draw / away encoded by pattern (solid / hairline barcode / outline), drawn in currentColor.
- Square everything: no radii, no drop shadows.
- Photography admitted only through a CSS Bayer dither; colour belongs to clubs.
- Signature move: the selected column wipes to white and inverts, then the detail band's club photos wipe from dither to colour.

## Colors

A strict monochrome: two absolutes, one readable grey, one structural grey. No accent.

### Neutral
- **Pitch Black** (ink): the page field, the inverted text colour inside selected and active elements, and the backing of selects and the highlights label.
- **Floodlight White** (paper): primary text, every data mark (bars, scatter points, barcode hits), the fill of selected columns, active tabs, active leagues, hit tags and the primary button.
- **Terrace Grey** (dim): the single secondary text step, 8.6:1 on black. Dates, units, table headers, captions, notes, inactive tabs and leagues, the "vs" separator.
- **Hard Rule** (rule): full-strength 1px hairlines that frame regions: header and league strip bottoms, field top and bottom, block-title underlines, the season rail's left edge, select borders.
- **Faint Rule** (rule-faint): 1px internal dividers only: between fixture columns, table rows, pipeline steps, league tabs; chart gridlines and rating baselines.

### Named Rules
**The Club Colour Rule.** The interface owns no hue. The only colour on any page is club colour: crests at all times, and club photography once revealed by selection or pointer. Only images verified as photographs are shown at all: TheSportsDB's crest wallpapers and templated match posters are excluded by the pipeline, and past meetings are drawn as score cards rather than pictures.

**The One Grey Rule.** Secondary text uses dim and nothing else. Faint rule is structure, never text; a third text grey or an opacity-tinted grey for new copy is out. (Inside fixture columns, secondary numbers use opacity 0.72 so they invert with the column; that is the only sanctioned opacity text.)

## Typography

**Display Font:** Martian Mono Variable (self-hosted via Fontsource, with ui-monospace, SFMono-Regular, Menlo, monospace)
**Body Font:** the same
**Label/Mono Font:** the same

**Character:** A single wide-stance monospace carrying everything from the wordmark to table footnotes. Tabular figures are on globally, so every number aligns in its column; the width axis does the work other systems give to a second face.

### Hierarchy
- **Display** (700, 2.5rem, 1.15, width 80%, -0.03em): page titles on long-read pages such as the Model page.
- **Headline** (700, 1.75rem, 1.15, width 80%, -0.02em): view titles ("Premier League Round 8", with the round in dim at 400) and the fixture detail title; drops to 1.375rem under 640px.
- **Title** (700, 1.125rem): scoreline figures, the model's headline metric in the season rail; also the Model page lede at 400.
- **Data** (400, 0.9375rem, width 80%, +0.02em): team short codes in columns, fact values, score cells, message text.
- **Body** (400, 0.8125rem, 1.5, tabular figures): default text and table cells.
- **Label** (600 or 700, 0.75rem, +0.06em to +0.08em, uppercase): tabs, league strip, table headers, block titles, row labels, tags, buttons. Block titles take 700 and +0.08em over a hard rule.

Headings h1-h3 are 700, line-height 1.15, balanced wrapping, -0.01em by default.

### Named Rules
**The One Face Rule.** Martian Mono is the only family. Hierarchy comes from size, weight (400, 600, 700, 800 for the wordmark only) and width, never from a second face.

**The Width Axis Rule.** Dense data (dates, codes, probabilities, xG, table numerics, titles) compresses with font-stretch 80% rather than dropping size. The wordmark alone stretches to 112.5%.

## Layout

A full-bleed page with a fluid gutter (clamp 16px to 32px) and a four-row shell: header (56px min), league strip, main, footer, each separated by a hard rule. Fixture and results views place a 300px season rail beside the main column with a 32px gap; the rail carries its own hard rule on the left and moves beneath the content, ruled on top, below 1100px.

The fixture field is a grid of equal columns (one per fixture, default ten), each at least 420px tall, framed top and bottom by hard rules and divided by faint rules. Below 760px columns become rows: date on the left, teams over a horizontal 14px bar in the middle, numbers on the right, xG hidden.

Spacing follows a 4px-based scale (4, 8, 12, 16, 24, 32, 48, 64). Inside data, 8 and 12 dominate; between sections, 32 to 64. Prose blocks are capped at 60 to 70ch. Wide tables scroll horizontally inside their own container, and on narrow screens non-standings tables collapse rows into two-column grids with headers hidden.

Breakpoints in use: 560, 640, 760, 900, 1000, 1100px.

## Elevation & Depth

Flat. There are no drop shadows and no layered surfaces; depth is replaced by inversion. A selected or active element becomes a white figure on the black field (fixture column, tab, league, model column header, hit tag, button), and everything inside it flips to black because marks are drawn in currentColor. The only box-shadow in the system is an inset 1px outline, used as a stroke (away bars, miss tags, crest fallbacks, legend swatches), not as elevation.

### Named Rules
**The Inversion Rule.** Emphasis and selection are expressed by swapping ink and paper, never by lifting, glowing or tinting.

## Shapes

Square, hairline, rectilinear. Radius is zero everywhere, including native selects (appearance reset, custom white triangle caret). Every line is 1px; outlines on focus are 2px currentColor offset 2px (inset 4px inside fixture columns). Icons are hairline SVGs on a 12px grid with a 1.25px stroke and square caps. Photographs sit in hard-edged frames at 16:9, or 21:9 in the detail band, with object-fit cover.

## Components

### Outcome Bar (signature)
The core data mark. Three segments in home / draw / away order, separated by a 2px gap (1px when compact), each sized by probability.
- **Home:** solid currentColor.
- **Draw:** hairline barcode, 1px line every 3px (horizontal lines in vertical bars, vertical lines in horizontal ones).
- **Away:** 1px inset outline in currentColor.
- **Sizes:** vertical fills the column (min 120px, 160px in fixture columns); horizontal is 20px tall, compact 8px, mobile 14px.
- A key (18 x 10px swatches with dim labels) accompanies any view showing bars.

### Fixture Column (signature)
A full-height button per fixture: date (label size, 80% width, 0.72 opacity), crest plus three-letter code for each side, vertical outcome bar, the 1 / X / 2 numbers with the pick in 700 at full opacity, then xG.
- **Hover:** a 3px white band wipes in at the top.
- **Selected:** a white fill wipes down from the top to 100% (220ms, ease-out-expo) and all text and marks invert to black.

### Dither Image (signature)
Club and match photography rendered as a 1-bit ordered dither in pure CSS: the colour image sits underneath; above it, a grayscale copy is overlay-blended with a 4x4 Bayer matrix tile and the group is pushed through grayscale(1) contrast(40). This works on cross-origin images with no canvas and no SVG feImage.
- **Hover / focus:** the dither layer fades out (220ms), revealing colour.
- **Revealed (selection):** the dither layer wipes away top to bottom via clip-path over 600ms, a beat after the column inversion. Only verified club photographs are ever rendered, so everything this reveals is club colour.
- **Fallback:** a faint-ruled empty frame, or the scoreline set large, keeps the layout when an image fails.

### Crest
Club crests at native colour, object-fit contain. When a crest is missing, a square inset-outlined box with the club's initials stands in.

### Navigation
- **Header tabs:** uppercase label type in dim; hover to white (160ms); the current page is a solid white block with black text running the full header height.
- **League strip:** a horizontally scrolling row of uppercase, 80%-width labels separated by faint rules; the active league is a white block.
- **Mobile:** below 900px the tabs wrap to their own scrolling row under a faint rule and the timestamp hides.

### Buttons
- **Shape:** square (0px).
- **Primary:** white fill, black uppercase label at 700, +0.04em, 8px 16px padding.
- **Hover:** a 1px white outline appears 2px outside the button. **Focus:** 2px currentColor outline offset 2px.
- Text links are underlined at 1px with a 0.25em offset, thickening to 2px on hover; "more" links pair uppercase label text with a hairline arrow.

### Inputs / Fields
- **Select:** black background, 1px white border, square, white triangle caret on the right, label-size text.
- **Focus:** the global 2px currentColor outline.

### Tables
Right-aligned numerics with tabular figures, 8px vertical padding, faint rules between rows, hard rule under dim uppercase headers. The model's own column is distinguished by inversion (white header, bold values). Row hover is a near-black tint.

### Tags
- **Hit:** white block, black uppercase 700 label.
- **Miss:** inset 1px white outline, white label. Same pattern logic as the outcome bar: solid means called, outline means missed.

### Charts
Season barcode and calibration chart follow the same grammar: faint gridlines and baseline, a dashed white diagonal for perfect calibration, solid white for the model or a hit, 1px outline for the market or a miss, dim 10px tick labels.

### Loading
A skeleton of ten faint-ruled empty columns with a hairline scanline that steps across in ten discrete jumps (1.6s, steps(10)).

## Do's and Don'ts

### Do:
- **Do** encode home, draw and away by pattern: solid, 1px-in-3px barcode, 1px outline, all in currentColor so they invert with their container.
- **Do** express selection and emphasis by inverting ink and paper.
- **Do** run all dense numbers through the 80% width axis with tabular figures.
- **Do** pass every photograph through the CSS Bayer dither; let club photos resolve to colour on selection with a top-to-bottom wipe.
- **Do** separate regions with 1px hard rules and internal rows with 1px faint rules.
- **Do** keep motion to wipes on --ease-out (cubic-bezier(0.16, 1, 0.3, 1)) at 160ms for colour changes and 220ms for wipes, and honour reduced motion.

### Don't:
- **Don't** introduce an interface accent colour or use colour to encode outcome, hit/miss or model versus market.
- **Don't** show crest wallpapers or templated match posters as imagery; the pipeline's photo check (flat-colour share and upload era) decides what counts as a photograph.
- **Don't** set text in rule-faint or add a second secondary grey.
- **Don't** round corners or add drop shadows; inset 1px strokes are the only box-shadow.
- **Don't** add a second typeface.
- **Don't** use SVG feImage or canvas for the dither; it fails in Chrome and on cross-origin images.
