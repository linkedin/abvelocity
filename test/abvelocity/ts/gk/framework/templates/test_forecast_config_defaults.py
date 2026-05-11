"""For more tests, see test_forecast_config.py"""

from abvelocity.ts.gk.framework.templates.autogen.forecast_config import ModelComponentsParam
from abvelocity.ts.gk.framework.templates.forecast_config_defaults import ForecastConfigDefaults


from abvelocity.ts.gk_test_gate import gk_test_gate

pytestmark = gk_test_gate


def test_apply_model_components_defaults():
    """Tests apply_model_components_defaults"""
    assert ForecastConfigDefaults().apply_model_components_defaults(None) == ModelComponentsParam()
    mcp = ModelComponentsParam({"growth": "growth"})
    assert ForecastConfigDefaults().apply_model_components_defaults(mcp) == mcp
    assert ForecastConfigDefaults().apply_model_components_defaults([mcp]) == mcp
    assert ForecastConfigDefaults().apply_model_components_defaults([None, mcp]) == [ModelComponentsParam(), mcp]


def test_apply_model_template_defaults():
    """Tests apply_model_template_defaults"""
    assert ForecastConfigDefaults().apply_model_template_defaults(model_template=None) == "AUTO"
    mt = "RANDOM_TEMPLATE"
    assert ForecastConfigDefaults().apply_model_template_defaults(model_template=mt) == mt
    assert ForecastConfigDefaults().apply_model_template_defaults(model_template=[mt]) == mt
    assert ForecastConfigDefaults().apply_model_template_defaults(model_template=[None, mt]) == ["AUTO", mt]
