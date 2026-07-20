"""离线行情阶段标注研究原型。

本包不会注册为交易策略，也不会接入现有回测流程。
"""

from .labeler import MarketRegimeLabeler, RegimeLabelerConfig, audit_price_data

__all__ = ["MarketRegimeLabeler", "RegimeLabelerConfig", "audit_price_data"]
