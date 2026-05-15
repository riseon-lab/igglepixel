# Moonstone UI Redesign Guide

## Direction

Replace the pixel/retro visual language with a clean modern interface. The app should feel calm, premium, legible, and consistent across every view: Models, Workspace, Running, Trainers, Trainer Monitor, LoRAs, Assets, Downloads, Settings, modals, lightbox, drawers, cards, and mobile nav.

## Typography

- Use Outfit as the primary font everywhere.
- Avoid pixel, mono-first, terminal-like, all-caps styling except for short metadata labels where it improves scanning.
- Headings should be readable and modern: medium weight, normal letter spacing, no blocky/pixel fonts.
- Body text should use comfortable line height and clear contrast.
- Monospace is allowed only for paths, filenames, command previews, logs, numeric diagnostics, and code-like data.

## Palette

- App background: moonstone grey, a cool blue-grey neutral.
- Main surfaces: slightly lighter moonstone panels.
- Elevated surfaces: soft, modern cards with subtle border and shadow.
- Buttons: dark moonstone fills with white text.
- Accent: solid baby blue. Use it for active nav, primary actions, selected states, progress, focus rings, and key highlights.
- Avoid magenta, neon cyan, lime, violet, scanline, grid, and pixel accents.

Suggested tokens:

- `--moon-bg`: `#5f7378`
- `--moon-bg-deep`: `#4e6368`
- `--moon-panel`: `#6f8388`
- `--moon-panel-2`: `#7d9095`
- `--moon-button`: `#31494f`
- `--moon-button-hover`: `#263d43`
- `--moon-border`: `rgba(255,255,255,0.22)`
- `--moon-border-strong`: `rgba(255,255,255,0.34)`
- `--moon-text`: `#ffffff`
- `--moon-text-soft`: `rgba(255,255,255,0.78)`
- `--moon-text-muted`: `rgba(255,255,255,0.58)`
- `--moon-accent`: `#9edcff`
- `--moon-accent-strong`: `#7fcfff`

## Shape And Depth

- Modern radius: 12px for cards/panels/modals, 10px for controls, 999px only for pills.
- Use subtle shadows, not harsh outlines.
- Use soft borders for separation.
- No scanlines, corner ticks, grid backgrounds, pixel shadows, chunky outlines, or blinking cursors.
- Cards should not feel nested inside other cards unless the hierarchy is functional.

## Controls

- Primary actions: baby blue background with dark moonstone text.
- Secondary actions: dark moonstone background with white text.
- Ghost buttons: transparent with a soft white border.
- Inputs and selects: darkened moonstone fill, white text, baby blue focus ring.
- Sliders and progress bars: baby blue fill, rounded tracks.
- Active tabs/nav/steps: baby blue accent, clear background, readable text.

## Layout

- Keep operational screens dense but breathable.
- Maintain strong hierarchy: page title, short supporting copy, filters/actions, content.
- Mobile views must avoid sticky controls that float over content unless absolutely necessary.
- Text must never rely on pixel-style compressed spacing to fit.
- Horizontal scrolling is acceptable only for tab strips and steppers.

## Implementation Notes

- Existing `.px-*` classes may remain in markup, but they must visually render as moonstone modern components.
- The final CSS override should sit near the end of `ui/styles.css` so it wins over the old pixel cascade.
- Remove the extra pixel font import from `ui/index.html`.
- Keep Outfit loaded.
- Preserve existing JS behavior and IDs.
- Verify at desktop width and mobile width for at least Models, Trainers, Trainer Running, LoRAs, Assets, Downloads, Settings, Workspace, HF modal, and Lightbox.
