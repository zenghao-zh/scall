from unittest import TestCase

from cyclonebasecall.evaluation.mapping_and_parsing import get_n_reads_results


class TestEvaluation(TestCase):
    def setUp(self) -> None:
        self.ref_path_0 = "/workspace/refs/ecoli.fasta"
        self.ref_path_1 = "/workspace/refs/Bacillus_subtilis.fasta"
        self.ref_path_2 = "/workspace/refs/Human.fasta"
        self.ref_path_3 = "/workspace/refs/rice.fasta"
        self.ref_path_4 = "/workspace/refs/BY4741.fasta"

    def test_eval1(self):
        fastq_path = "/workspace/datasets/tmp/test.fastq"
        res = get_n_reads_results(fastq_path, self.ref_path_0)
        for key, val in res.items():
            print("{}: {}".format(key, val))
