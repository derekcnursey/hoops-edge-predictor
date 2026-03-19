import { GetServerSideProps } from "next";
import MarchHub from "../components/march/MarchHub";
import { loadMarchPageProps, MarchPageProps } from "../lib/bracket/marchPageData";

export const getServerSideProps: GetServerSideProps<MarchPageProps> = async () => {
  return { props: await loadMarchPageProps() };
};

export default function MarchPage(props: MarchPageProps) {
  return <MarchHub {...props} />;
}
