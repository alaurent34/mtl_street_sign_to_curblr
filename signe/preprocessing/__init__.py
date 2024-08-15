from signe.preprocessing.signalec.signalec_preprocessing import (
    transform_signalec as process_signalec
)
from signe.preprocessing.paid_parking_preprocessing import (
    main as process_mtl_paid_parking
)
from signe.preprocessing.catalogue_preprocessing import (
    main as process_catalog
)
from signe.preprocessing.fire_hydrants import process_fire_hydrants


__all__ = ['process_signalec', 'process_mtl_paid_parking', 'process_catalog',
           'process_fire_hydrants']
