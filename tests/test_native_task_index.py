"""Native coarse/fine task ids must select the coarse instruction."""

import ast
from pathlib import Path
import unittest

import torch


# Load the production helper without initializing unrelated training packages.
_PATH = Path(__file__).resolve().parents[1] / "lingbotvla/data/vla_data/base_dataset.py"
_TREE = ast.parse(_PATH.read_text())
_TREE.body = [node for node in _TREE.body if isinstance(node, ast.FunctionDef)
              and node.name in {"_coarse_task_index", "_get_task_name"}]
_NAMESPACE = {"torch": torch}
exec(compile(_TREE, str(_PATH), "exec"), _NAMESPACE)
coarse_task_index = _NAMESPACE["_coarse_task_index"]
get_task_name = _NAMESPACE["_get_task_name"]


class NativeTaskIndexTest(unittest.TestCase):
    def test_scalar_and_singleton_backwards_compatibility(self):
        for value in (0, torch.tensor(3), torch.tensor([4])):
            self.assertEqual(coarse_task_index(value), int(torch.as_tensor(value).item()))

    def test_native_pair_uses_coarse_instruction(self):
        tasks = {0: "Put the drinks into the refrigerator"}
        task_id = coarse_task_index(torch.tensor([0, 1830]))
        self.assertEqual(task_id, 0)
        self.assertEqual(get_task_name(tasks, task_id), tasks[0])

    def test_reject_unrecognized_shapes(self):
        for value in (torch.tensor([]), torch.tensor([1, 2, 3]), torch.tensor([[1, 2]])):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                coarse_task_index(value)

    def test_reject_noninteger_identifiers(self):
        for value in (torch.tensor(1.5), torch.tensor([0.0, 1.0]), torch.tensor(True)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                coarse_task_index(value)


if __name__ == "__main__":
    unittest.main()
