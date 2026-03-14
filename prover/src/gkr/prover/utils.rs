use crate::merkle_trees::MerkleTreeCapVarLength;
use crate::merkle_trees::MerkleTreeConstructor;

pub fn flatten_merkle_caps_iter_into(
    tree_caps_iter: impl Iterator<Item = MerkleTreeCapVarLength>,
    dst: &mut Vec<u32>,
) {
    for cap in tree_caps_iter {
        cap.add_into_buffer(dst);
    }
}

pub fn flatten_merkle_caps_into<T: MerkleTreeConstructor>(trees: &[T], dst: &mut Vec<u32>) {
    for subtree in trees.iter() {
        for cap_element in subtree.get_cap().cap.iter() {
            dst.extend_from_slice(cap_element);
        }
    }
}

pub fn flatten_merkle_caps<T: MerkleTreeConstructor>(trees: &[T]) -> Vec<u32> {
    let mut result = vec![];
    flatten_merkle_caps_into(trees, &mut result);

    result
}
