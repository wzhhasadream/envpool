# Pinned Gymnasium Robotics assets

These assets target Farama's Gymnasium Robotics v1.4.2, official commit
`42eae53b2b27321d29090c219ce2f675c596de77`:
https://github.com/Farama-Foundation/Gymnasium-Robotics/tree/42eae53b2b27321d29090c219ce2f675c596de77

The repository rule downloads the immutable commit archive:
https://github.com/Farama-Foundation/Gymnasium-Robotics/archive/42eae53b2b27321d29090c219ce2f675c596de77.tar.gz

Its SHA256 is
`3c675a90d800b7f1a48710051f106d6c4019f06d859aa928a5cd82e8be141890`.
The strip prefix retains only `gymnasium_robotics/envs` from that archive.
The upstream version and existing asset selection are unchanged.

On 2026-10-19, the official Git remote's `refs/tags/v1.4.2` independently
resolved to the commit above. All 390 archive files, including the 257 files
under `gymnasium_robotics/envs`, were compared byte-for-byte against a checkout
of that commit and verified against their Git blob IDs. Extracted files were
checked again; there were no missing, extra, or changed files, and executable
modes matched. Git metadata and local untracked Bazel BUILD/WORKSPACE glue
were excluded from the comparison.

The previous tag-archive rule expected SHA256
`0c47791b0d8aa167f63a574793f1c54724b545ae2b03225c78946139f2a4c8ac`,
but both official tag-archive URLs returned
`e5f49da77b31c3f7be02eea7f9788d7fd817d4eccdcef625b1a85a77e221532f`
during validation. The replacement pins the verified official commit and its
independently checked contents rather than accepting the mismatching tag
archive by changing its checksum alone.
