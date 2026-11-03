import mill._
import scalalib._
import scalafmt._
import os.Path
import publish._
import $file.`rocket-chip`.common
import $file.`rocket-chip`.common
import $file.`rocket-chip`.cde.common
import $file.`rocket-chip`.hardfloat.common

val defaultScalaVersion = "2.13.15"

def defaultVersions = Map(
  "chisel"        -> mvn"org.chipsalliance::chisel:7.13.0",
  "chisel-plugin" -> mvn"org.chipsalliance:::chisel-plugin:7.13.0",
)

val pwd = os.Path(sys.env("MILL_WORKSPACE_ROOT"))

trait HasChisel extends ScalaModule {
  def chiselModule: Option[ScalaModule] = None

  def chiselPluginJar: T[Option[PathRef]] = None

  def chiselIvy: Option[Dep] = Some(defaultVersions("chisel"))

  def chiselPluginIvy: Option[Dep] = Some(defaultVersions("chisel-plugin"))

  def sourcecodeIvy = mvn"com.lihaoyi::sourcecode:0.4.4"

  override def scalaVersion = defaultScalaVersion

  override def scalacOptions = super.scalacOptions() ++
    Agg("-language:reflectiveCalls", "-Ymacro-annotations", "-Ytasty-reader")

  override def ivyDeps = super.ivyDeps() ++ Agg(chiselIvy.get) ++ Agg(sourcecodeIvy)

  override def scalacPluginIvyDeps = super.scalacPluginIvyDeps() ++ Agg(chiselPluginIvy.get)
}

object rocketchip extends `rocket-chip`.common.RocketChipModule with HasChisel {

  val rcPath = pwd / "rocket-chip"
  override def millSourcePath = rcPath

  def mainargsIvy = mvn"com.lihaoyi::mainargs:0.7.0"

  def json4sJacksonIvy = mvn"org.json4s::json4s-jackson:4.0.7"

  object macros extends `rocket-chip`.common.MacrosModule with HasChisel {
    def scalaReflectIvy = mvn"org.scala-lang:scala-reflect:${scalaVersion}"
  }

  object cde extends `rocket-chip`.cde.common.CDEModule with HasChisel {
    override def millSourcePath = rcPath / "cde" / "cde"
  }

  object hardfloat extends `rocket-chip`.hardfloat.common.HardfloatModule with HasChisel {
    override def millSourcePath = rcPath / "hardfloat" / "hardfloat"
  }

  def macrosModule = macros

  def hardfloatModule = hardfloat

  def cdeModule = cde

}

object utility extends SbtModule with HasChisel {
  override def millSourcePath = pwd

  override def moduleDeps = super.moduleDeps ++ Seq(rocketchip)

  override def scalacOptions = super.scalacOptions() ++ Agg("-deprecation", "-feature")
}
