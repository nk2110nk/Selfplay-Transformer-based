from sao.my_utilities import from_xml_str, luaf_call, muf_call
from sao.my_negotiators import reset
from negmas.utilities import UtilityFunction, LinearUtilityAggregationFunction, MappingUtilityFunction
from negmas.sao import SAONegotiator

# UtilityFunction.from_xml_str = from_xml_str
# LinearUtilityAggregationFunction.__call__ = luaf_call
# MappingUtilityFunction.__call__ = muf_call
SAONegotiator.reset = reset
