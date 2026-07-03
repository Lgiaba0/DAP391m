---
name: Orbital Glass
colors:
  surface: '#131313'
  surface-dim: '#131313'
  surface-bright: '#393939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1b1b1b'
  surface-container: '#1f1f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353535'
  on-surface: '#e2e2e2'
  on-surface-variant: '#cfc4c5'
  inverse-surface: '#e2e2e2'
  inverse-on-surface: '#303030'
  outline: '#988e90'
  outline-variant: '#4c4546'
  surface-tint: '#c6c6c6'
  primary: '#c6c6c6'
  on-primary: '#303030'
  primary-container: '#000000'
  on-primary-container: '#757575'
  inverse-primary: '#5e5e5e'
  secondary: '#c6c6c7'
  on-secondary: '#2f3131'
  secondary-container: '#454747'
  on-secondary-container: '#b4b5b5'
  tertiary: '#c6c6c6'
  on-tertiary: '#303030'
  tertiary-container: '#000000'
  on-tertiary-container: '#757575'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e2e2e2'
  primary-fixed-dim: '#c6c6c6'
  on-primary-fixed: '#1b1b1b'
  on-primary-fixed-variant: '#474747'
  secondary-fixed: '#e2e2e2'
  secondary-fixed-dim: '#c6c6c7'
  on-secondary-fixed: '#1a1c1c'
  on-secondary-fixed-variant: '#454747'
  tertiary-fixed: '#e2e2e2'
  tertiary-fixed-dim: '#c6c6c6'
  on-tertiary-fixed: '#1b1b1b'
  on-tertiary-fixed-variant: '#474747'
  background: '#131313'
  on-background: '#e2e2e2'
  surface-variant: '#353535'
  spectrum-cyan: '#00F0FF'
  spectrum-purple: '#8B5CF6'
  spectrum-pink: '#EC4899'
  spectrum-blue: '#3B82F6'
  glass-stroke: rgba(255, 255, 255, 0.15)
  deep-space: '#050505'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 72px
    fontWeight: '700'
    lineHeight: 80px
    letterSpacing: -0.04em
  headline-lg:
    fontFamily: Inter
    fontSize: 40px
    fontWeight: '600'
    lineHeight: 48px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 38px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.1em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  container-max: 1440px
  gutter: 24px
  margin-desktop: 80px
  margin-mobile: 20px
  stack-sm: 8px
  stack-md: 16px
  stack-lg: 32px
---

## Brand & Style

The design system is centered on a "Post-Planetary Luxury" aesthetic, blending the stark, utilitarian precision of aerospace engineering with the fluid, organic vibrancy of modern AI. The target audience consists of high-net-worth travelers and tech-forward explorers who seek a frictionless, immersive booking experience.

The visual style is a sophisticated hybrid of **Glassmorphism** and **High-Contrast Minimalism**. It utilizes deep space blacks to establish infinite depth, layered with ultra-refined "Liquid Glass" surfaces. These surfaces are not merely transparent; they act as optical lenses that refract the vibrant "Gemini Spectrum" accents, creating a sense of movement and intelligence. The emotional response is one of awe, reliability, and futuristic comfort.

## Colors

The palette is anchored in a true-black foundation (`#000000`), drawing directly from the void of space to maximize the contrast of the glass elements. The primary interaction color is pure white, ensuring clinical legibility against dark backgrounds.

The "Gemini Spectrum" is used exclusively for atmospheric glows, data visualizations, and active AI states. Rather than static fills, these colors should be applied as linear gradients (e.g., Cyan to Purple) or as soft, blurred background "blobs" that sit behind the frosted glass layers. This creates a chromatic depth that feels holographic rather than flat.

## Typography

This design system utilizes **Inter** for all primary communication to maintain a neutral, high-end Swiss feel. It relies on tight letter-spacing and varied weights to establish hierarchy. To nod toward the technical nature of space travel, **JetBrains Mono** is introduced for secondary labels, coordinates, and data points, providing a precise, "instrument panel" feel.

Display type should be set with negative tracking to feel dense and impactful. Body text requires generous line height (1.5x) to ensure readability against dynamic, blurred backgrounds.

## Layout & Spacing

The layout follows a **Fixed Grid** model for desktop, centered within a 1440px container to evoke the feeling of a wide-angle viewport. A 12-column system is used with wide gutters to allow the background Earth imagery to breathe through the gaps between glass modules.

On mobile, the layout shifts to a single-column fluid flow with tight 20px margins. Content cards should utilize "Safe Areas" to prevent text from overlapping the curved edges of the glass containers. Vertical rhythm is strictly enforced in increments of 8px to maintain technical precision.

## Elevation & Depth

Depth is achieved through **Backdrop Refraction** rather than traditional shadows. 
1. **Base Layer:** The immersive Earth/Space imagery.
2. **Atmospheric Layer:** Vibrant spectrum glows (`spectrum-cyan`, `spectrum-pink`) with 100px+ Gaussian blurs.
3. **Glass Layer:** Surfaces use a 20px to 40px backdrop-blur. The fill is a semi-transparent white (e.g., `rgba(255, 255, 255, 0.03)`).
4. **Edge Layer:** A 1px solid "inner-glow" stroke (`glass-stroke`) defines the boundary of the glass, catching the light like the edge of a lens.

Avoid drop shadows; use "Outer Glows" sparingly to indicate active AI interaction states.

## Shapes

The design system employs a **Rounded** (0.5rem base) geometry. While the brand is futuristic, perfectly sharp corners feel overly aggressive; the soft rounding mimics the precision-milled edges of aerospace glass and the curvature of the Earth's horizon. Smaller components like chips and toggles may use higher roundedness (pill-shaped) to distinguish them as touchable, interactive elements.

## Components

- **Buttons:** Primary buttons are solid white with black text for maximum "pop." Secondary buttons are "Ghost" style with the `glass-stroke` border and a subtle hover refraction effect.
- **Glass Cards:** These are the primary layout containers. They must feature `backdrop-filter: blur(24px)` and a subtle linear gradient border (top-left to bottom-right) to simulate light hitting the edge.
- **AI Accents:** Use a "Glow Bar"—a 2px thick line using a Gemini Spectrum gradient—at the top of cards or inputs when the AI is processing or active.
- **Inputs:** Fields are fully transparent with only a bottom border or a very faint glass fill. On focus, the border should transition to a `spectrum-cyan` glow.
- **Data Visuals:** Use monospaced labels for all numerical data. Charts should use the Spectrum palette with thin, glowing lines and no fills to maintain the "holographic" aesthetic.
- **Immersive Hero:** The hero section must feature a slow-parallax Earth overview. Text should be placed in high-contrast areas (the black of space) to ensure legibility.