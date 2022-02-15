shst match output/signs_preprocessed.geojson --join-points --join-point-sequence-field=chainage \
                                                           --join-points-match-fields=CODE_RPA,PANNEAU_ID_RPA \
                                                           --snap-intersections --trim-intersections-radius=10 \
                                                           --search-radius=15 \
                                                           --out=output/shst/signs_preprocessed.geojson 

shst match output/paid_parking_preprocessed.geojson  --buffer-points --buffer-points-length=7 \
                                                     --buffer-merge --buffer-merge-match-fields=SK_D_Troncon,No_Place \
                                                     --buffer-merge-group-fields=SK_D_Troncon,Tarif_Hr \
                                                     --snap-intersections --trim-intersections-radius=10 \
                                                     --out=output/shst/paid_parking_preprocessed.geojson

shst match output/hydrants_preprocessed.geojson  --buffer-points --buffer-points-length=6 \
                                                 --snap-intersections --trim-intersections-radius=10 \
                                                 --out=output/shst/hydrants_preprocessed.geojson
