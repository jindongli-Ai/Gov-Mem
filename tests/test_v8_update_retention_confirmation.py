import pytest
from scripts.run_v8_update_retention_confirmation import choose,disk2


def test_sampling_excludes_exposed_and_is_order_independent():
    ids=['a','b','c','d','e','f']
    chosen=choose(ids,{'b','d'},'domain',count=3)
    assert len(chosen)==3 and not set(chosen)&{'b','d'}
    assert chosen==choose(list(reversed(ids)),{'d','b'},'domain',count=3)
    with pytest.raises(ValueError):choose(ids,set(ids),'domain')


def test_writable_paths_must_resolve_to_disk2(tmp_path):
    assert str(disk2('/mnt/data_disk_2/fuyali/codes')).startswith('/mnt/data_disk_2/')
    with pytest.raises(ValueError):disk2('/mnt/data_disk/home/fuyali/codes')
    link=tmp_path/'old_disk';link.symlink_to('/mnt/data_disk')
    with pytest.raises(ValueError):disk2(link/'outputs')
