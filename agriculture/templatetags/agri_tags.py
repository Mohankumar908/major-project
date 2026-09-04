from django import template

register = template.Library()

CROP_ICONS = {
    'Green Gram': '🫘',
    'Rice':       '🌾',
    'Wheat':      '🌿',
    'Tomato':     '🍅',
    'Maize':      '🌽',
    'Potato':     '🥔',
    'Soybean':    '🫘',
    'Sugarcane':  '🎋',
    'Cotton':     '☁️',
    'Other':      '🌱',
}

@register.filter
def crop_icon(crop_name):
    """Return the emoji for a crop name, defaulting to 🌱."""
    return CROP_ICONS.get(crop_name, '🌱')

@register.filter
def health_badge_class(score):
    """Return the badge CSS class for a health score."""
    if score is None:
        return 'badge-slate'
    if score >= 75:
        return 'badge-green'
    if score >= 50:
        return 'badge-amber'
    return 'badge-red'
