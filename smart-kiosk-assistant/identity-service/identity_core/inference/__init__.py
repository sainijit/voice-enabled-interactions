"""Inference layer: OpenVINO face + voice embedding engines.

Populated in Phase 4 using the dlstreamer/OpenVINO base image.  Engines are
constructed via a factory and selected through a strategy interface so backends
can be swapped via configuration.
"""
