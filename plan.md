# Citivia Studio - MVP Specification

## Overview

Build a self-hosted AI image generation platform designed for deployment on RunPod.

The system should provide a modern web interface for managing models, LoRAs, assets, and image generation workflows while maintaining strong security and isolation between components.

The initial goal is to support:

* Qwen 2512 (Base Generation)
* Qwen Edit 2511 (Image Editing)

The platform must be mobile-responsive and designed for future expansion into training, datasets, and additional models.

---

Everything should save to /workspace only

# Infrastructure

## Deployment

Target deployment:

* RunPod
* RTX PRO 6000 (96GB VRAM)
* L40S (48GB VRAM)

Requirements:

* Docker container deployment
* Git repository pulled automatically on startup
* UI exposed on Port 3000
* Ability to pull latest UI updates from Git without requiring full pod rebuilds

---

# Security

## Encryption

All user assets must be encrypted at rest.

This includes:

* Uploaded images
* Reference images
* Generated images
* LoRA files
* Temporary outputs
* Cached assets

Requirements:

* AES-256-GCM (recommended industry standard)
* Server stores encrypted files only
* Decryption occurs client-side
* Browser worker handles decryption
* Assets decrypted only in memory
* No decrypted files written to disk

Validation:

* Test uploads
* Test downloads
* Test previews
* Test generation outputs
* Confirm no visual corruption or rendering artefacts

---

# Authentication

On first launch:

* Present setup screen
* Create username and password

Session requirements:

* Single active session per account
* Logging in on a second browser/device invalidates previous session
* Secure session tokens
* Logout functionality

---

# Model Architecture

Each model must run in its own isolated environment.

Requirements:

* Independent runners
* Dependency isolation
* No shared Python environments
* No package conflicts
* Ability to stop/start models individually

Future models should be installable without affecting existing deployments.

---

# LoRA Management

## Civitai Integration

Users can:

* Enter Civitai API key
* Save API key securely
* Download LoRAs using Civitai URLs
* Store LoRAs in dedicated LoRA directory

## Hugging Face Integration

Users can:

* Enter Hugging Face API key
* Download LoRAs
* Browse downloaded LoRAs
* Use LoRAs within workflows

Future:

* Upload trained LoRAs to Hugging Face

---

# Navigation

Use a collapsible sidebar.

Desktop:

* Left sidebar

Mobile:

* Bottom navigation bar

Navigation items:

* Running
* Models
* Assets
* LoRAs
* Downloads
* Data Studio (future)
* Training (future)
* Settings

---

# Pages

## Running

Display active models.

Features:

* Start model
* Stop model
* View status
* Resource usage

---

## Models

Card-based model selection.

Initial models:

* Qwen 2512
* Qwen Edit 2511

Future models can be added without redesign.

---

## Assets

Features:

* Upload images
* Upload references
* View uploads
* Download assets
* Delete assets

---

## LoRAs

Features:

* List installed LoRAs
* Upload LoRA
* Download from Civitai
* Download from Hugging Face
* Delete LoRA
* Future export to Hugging Face

---

## Downloads

Manage:

* Model downloads
* LoRA downloads
* Download history
* Download progress

---

## Settings

Configuration:

* Civitai API Key
* Hugging Face API Key
* Pull Latest Git Repository
* User Session Settings

---

# Generation Interface

## Qwen 2512

### Prompt

Large prompt textbox.

### Negative Prompt

Pre-populated default negative prompt.

User may edit.

### Resolution Presets

Visual selectable cards showing actual aspect ratio.

Include:

* 512 × 512
* 1024 × 1024
* 1380 × 1380
* 2048 × 2048
* 1344 × 1792
* 1024 × 1360
* 1080 × 1920
* 1920 × 1080

### Generation Controls

Steps:

* Default: 20
* Maximum: 200

Seed:

* Random
* User specified
* Reuse previous seed

CFG Scale:

* Default: 4
* Range: 0-100

Controls:

* Generate
* Add To Queue
* Stop
* Cancel

---

# Generation Preview

Live generation progress.

Requirements:

* Show intermediate previews during generation
* Update as steps progress
* Automatically focus latest generated image

Actions:

* View Full Size
* Download
* Delete
* Reuse Settings

Image containers must preserve aspect ratio and resolution without cropping.

---

# Queue

Display:

* Pending jobs
* Running jobs
* Completed jobs

Include:

* Thumbnail preview
* Progress indicators
* Status
* Download
* Delete
* View Full Size

---

# Qwen Edit 2511

Mirror the Qwen 2512 interface.

Additional requirement:

Reference Image upload area above prompt box.

Workflow:

Reference Image
↓
Prompt
↓
Generation Controls
↓
Generate

---

# Mobile Requirements

Responsive across:

* Mobile phones
* Tablets
* Desktop

Requirements:

* Sidebar becomes bottom navigation
* No header overlap
* No clipped controls
* No cropped image previews
* Touch-friendly controls

---

# Preview Build

Before connecting real models:

Create a fully functional UI prototype using mock data.

The preview build should allow:

* Navigation testing
* Session testing
* Queue testing
* Upload testing
* Layout validation
* Mobile validation

No real inference required.

The goal is to validate UX and workflow before integrating actual model runners.

---

# MVP Success Criteria

The MVP is complete when:

* User can login securely
* Models can be started/stopped
* Assets remain encrypted at rest
* LoRAs can be downloaded from Civitai and Hugging Face
* Qwen 2512 generation UI is operational
* Qwen Edit 2511 UI is operational
* Queue system works
* Mobile experience works cleanly
* RunPod deployment succeeds on RTX PRO 6000 and L40S
* Preview build is available before model integration




# Design Language

## Design Goals

The UI should feel:

* Premium
* Creative
* Modern
* Fast
* Professional

Avoid:

* Cyberpunk aesthetics
* Excessive gradients
* Tiny text
* Cluttered dashboards
* Generic admin-panel styling

The application should feel closer to a professional creative tool than a hobbyist AI interface.

---

# Colour Palette

## Primary Brand Colour

Lilac

```css
#A97DE0
```

RGB

```css
rgb(169, 125, 224)
```

Used for:

* Primary buttons
* Active navigation items
* Progress indicators
* Selected cards
* Focus states
* Sliders
* Toggles

---

## Dark Lilac

```css
#7852B0
```

Used for:

* Hover states
* Active button states
* Selected tabs
* Secondary accents

---

## Background

```css
#121212
```

Primary application background.

---

## Surface

```css
#1A1A1A
```

Cards and panels.

---

## Surface Hover

```css
#222222
```

Interactive hover state.

---

## Border

```css
#2D2D2D
```

Used sparingly for separation.

---

## Text

Primary:

```css
#FFFFFF
```

Secondary:

```css
#B0B0B0
```

Muted:

```css
#808080
```

---

# Typography

Minimum text size:

```css
16px
```

Never use text below 16px for inputs.

This prevents unwanted zooming on iPhone Safari.

Suggested scale:

```css
Body: 16px
Label: 16px
Section Title: 20px
Page Title: 28px
Hero Title: 36px
```

Font Family:

```css
Inter, sans-serif
```

or

```css
Geist, sans-serif
```

Font Weight:

```css
Regular: 400
Medium: 500
SemiBold: 600
Bold: 700
```

---

# Border Radius

Use a consistent rounded style.

```css
12px
```

for controls.

```css
16px
```

for cards.

```css
24px
```

for large panels.

Avoid sharp corners.

---

# Buttons

Primary:

* Lilac background
* White bold text
* Medium shadow
* Rounded corners

Secondary:

* Dark surface background
* Border
* White text

Danger:

* Dark red
* White text

Button height:

```css
48px
```

Minimum.

---

# Layout

Use generous spacing.

```css
8px
16px
24px
32px
```

spacing scale.

Never cram controls together.

Prioritise readability and touch interaction.

---

# Navigation

Desktop:

* Left sidebar
* Collapsible

Mobile:

* Bottom navigation bar
* Fixed position

Active item:

* Lilac background
* White icon
* White text

Inactive item:

* Grey icon
* Grey text

---

# Cards

All major functionality should exist inside cards.

Card Style:

```css
background: #1A1A1A;
border: 1px solid #2D2D2D;
border-radius: 16px;
padding: 24px;
```

---

# Image Display

Generated images should always preserve their original aspect ratio.

Never crop previews.

Support:

* Portrait
* Landscape
* Square

Image containers should scale responsively while maintaining the correct dimensions.

---

# Animations

Use subtle animations only.

Examples:

* Fade transitions
* Progress updates
* Hover elevation
* Button state changes

Avoid:

* Flashing effects
* Large motion
* Excessive animations

The artwork should remain the centre of attention.

---

# General Rule

If a design decision is unclear, optimise for:

1. Simplicity
2. Readability
3. Touch friendliness
4. Creative workflow efficiency
5. Consistency

The interface should feel like a professional desktop application that also happens to work perfectly on mobile.
