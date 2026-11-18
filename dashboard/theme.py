"""
dashboard/theme.py
==================
Shared styling, color palettes, and component helpers for Rajasthan Royals
Pressure Performance Intelligence Dashboard.
"""

import json
from html import escape as escape_html

import streamlit as st
import streamlit.components.v1 as components

# Rajasthan Royals inspired palette
RR_BLUE = "#001D4A"
RR_ROYAL_BLUE = "#004B8C"
RR_PINK = "#E81B6C"
RR_ACCENT_PINK = "#FF4081"
RR_GOLD = "#E0B974"
RR_LIGHT_BG = "#F8F9FA"
RR_DARK_TEXT = "#1C252C"

CUSTOM_CSS = """
<style>
    /* Main branding accents */
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #001D4A;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #E81B6C;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #001D4A 0%, #004B8C 100%);
        border-radius: 12px;
        padding: 18px 22px;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 29, 74, 0.15);
        margin-bottom: 1rem;
    }
    .metric-card-pink {
        background: linear-gradient(135deg, #C2185B 0%, #E81B6C 100%);
        border-radius: 12px;
        padding: 18px 22px;
        color: white;
        box-shadow: 0 4px 12px rgba(232, 27, 108, 0.2);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        opacity: 0.85;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 4px;
        margin-bottom: 2px;
    }
    .metric-subtext {
        font-size: 0.8rem;
        opacity: 0.9;
    }
    .badge-hp {
        background-color: #E81B6C;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-normal {
        background-color: #004B8C;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
"""

def apply_theme():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def melt_glitch_text(
    text: str,
    subtitle: str = "",
    font_size: int = 36,
    font_weight: int = 800,
    color: str = RR_BLUE,
    cyan_color: str = "#5ce1ff",
    magenta_color: str = RR_PINK,
    radius: int = 110,
    drop_amount: int = 16,
):
    settings = json.dumps({
        "text": text,
        "fontSize": font_size,
        "fontWeight": font_weight,
        "color": color,
        "cyanColor": cyan_color,
        "magentaColor": magenta_color,
        "radius": radius,
        "dropAmount": drop_amount,
    }).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    fallback = escape_html(text)
    safe_subtitle = escape_html(subtitle)
    height = font_size + int(drop_amount * 2.6) + 16
    html = f"""
    <html>
      <head>
        <style>
          html, body {{ margin: 0; padding: 0; overflow: hidden; }}
          .header {{ font-family: Inter, sans-serif; }}
          .glitch-wrap {{
            display: inline-block;
            color: {color};
            font-size: {font_size}px;
            font-weight: {font_weight};
            line-height: 1.15;
            white-space: nowrap;
          }}
          .subtitle {{
            margin-top: 8px;
            margin-bottom: 12px;
            color: {RR_PINK};
            font-size: 1.1rem;
            font-weight: 600;
          }}
          .char {{ display: inline-block; position: relative; }}
          .base {{ position: relative; z-index: 3; display: inline-block; }}
          .cyan, .magenta {{ position: absolute; left: 0; top: 0; opacity: 0; }}
          .cyan {{ z-index: 1; filter: blur(0.5px); }}
          .magenta {{ z-index: 2; filter: blur(0.5px); }}
        </style>
      </head>
      <body>
        <div class="header">
          <div class="glitch-wrap" role="heading" aria-level="1" aria-label="{fallback}">{fallback}</div>
          <div class="subtitle">{safe_subtitle}</div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
        <script>
          const settings = {settings};
          const wrap = document.querySelector(".glitch-wrap");
          const chars = [];
          let setters = [];
          let mouseX = -9999;
          let active = false;
          let rafId;

          if (window.gsap && wrap) {{
            wrap.textContent = "";
            for (const character of settings.text) {{
              const holder = document.createElement("span");
              holder.className = "char";
              holder.style.fontSize = `${{settings.fontSize}}px`;
              holder.style.fontWeight = String(settings.fontWeight);

              const makeLayer = (className, layerColor) => {{
                const layer = document.createElement("span");
                layer.className = className;
                layer.style.color = layerColor;
                layer.textContent = character === " " ? "\\u00a0" : character;
                return layer;
              }};

              const cyan = makeLayer("cyan", settings.cyanColor);
              const magenta = makeLayer("magenta", settings.magentaColor);
              const base = makeLayer("base", settings.color);
              holder.append(cyan, magenta, base);
              wrap.appendChild(holder);
              chars.push({{ holder, base, cyan, magenta }});
            }}

            setters = chars.map((character) => ({{
              baseY: gsap.quickTo(character.base, "y", {{ duration: 0.45, ease: "power3.out" }}),
              cyanY: gsap.quickTo(character.cyan, "y", {{ duration: 0.65, ease: "power3.out" }}),
              cyanX: gsap.quickTo(character.cyan, "x", {{ duration: 0.6, ease: "power3.out" }}),
              magY: gsap.quickTo(character.magenta, "y", {{ duration: 0.8, ease: "power3.out" }}),
              magX: gsap.quickTo(character.magenta, "x", {{ duration: 0.75, ease: "power3.out" }}),
              cyanOpacity: gsap.quickTo(character.cyan, "opacity", {{ duration: 0.35 }}),
              magOpacity: gsap.quickTo(character.magenta, "opacity", {{ duration: 0.35 }}),
            }}));

            const handleMove = (event) => {{
              mouseX = event.clientX;
              active = true;
            }};
            const handleLeave = () => {{ active = false; }};
            wrap.addEventListener("mousemove", handleMove);
            wrap.addEventListener("mouseleave", handleLeave);

            const tick = () => {{
              chars.forEach((character, index) => {{
                const rect = character.holder.getBoundingClientRect();
                const distance = mouseX - (rect.left + rect.width / 2);
                const proximity = active
                  ? Math.max(0, 1 - Math.abs(distance) / settings.radius)
                  : 0;
                const eased = Math.sin(proximity * Math.PI * 0.5);
                const side = distance < 0 ? 1 : -1;
                const setter = setters[index];

                setter.baseY(eased * settings.dropAmount);
                setter.cyanY(eased * settings.dropAmount * 1.9);
                setter.cyanX(side * eased * 4);
                setter.magY(eased * settings.dropAmount * 2.6);
                setter.magX(-side * eased * 4);
                setter.cyanOpacity(eased * 0.9);
                setter.magOpacity(eased * 0.9);
              }});
              rafId = requestAnimationFrame(tick);
            }};
            rafId = requestAnimationFrame(tick);
          }}
        </script>
      </body>
    </html>
    """
    components.html(html, height=height, scrolling=False)


def render_header(title: str, subtitle: str):
    apply_theme()
    melt_glitch_text(title, subtitle)

def metric_card(title: str, value: str, subtext: str = "", pink: bool = False):
    card_class = "metric-card-pink" if pink else "metric-card"
    html = f"""
    <div class="{card_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
