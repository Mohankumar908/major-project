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


@register.filter
def format_disease(raw_name):
    """Format dataset disease class name for clean presentation."""
    if not raw_name:
        return 'Unprocessed'
    if '___' in raw_name:
        plant, disease = raw_name.split('___', 1)
        p_clean = plant.replace('_', ' ').replace('(', '').replace(')', '').replace(',', '').strip()
        d_clean = disease.replace('_', ' ').replace('(', '').replace(')', '').strip()
        if d_clean.lower() == 'healthy':
            return f"{p_clean} - Healthy"
        return f"{p_clean} - {d_clean}"
    return raw_name.replace('_', ' ')

