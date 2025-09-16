
for i in {0..31}; do
  fun job run \
    --gpu 1 \
    --command "python -m evaluation.evaluate" \
    --project sota  --name="evalchunk$i" \
    --verbose \
    -- \
    --config_name=release \
    --num_chunks=32 \
    --chunk_id="$i" \
    --overwrites model.kwargs.variant=ftm_tabicl_v0 name="ftm_tabicl_v0_release_1" 
done
